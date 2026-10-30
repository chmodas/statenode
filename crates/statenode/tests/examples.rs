//! Happy-path examples of the public API, written to be read.

use std::collections::BTreeSet;
use std::mem;
use std::sync::Arc;

use statenode::{
    BoxError, ChannelSpec, END, GraphBuilder, NameError, NodeContext, ReduceError, RunConfig,
    RunError, START, State, TaskPath, node, reducer,
};

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{Counter, CounterUpdate};

/// A hand-written state with two channels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Document {
    /// The document's title.
    title: String,
    /// How many times it was revised.
    revisions: u32,
}

/// An update to [`Document`]; `None` leaves a channel unwritten.
#[derive(Debug, Default)]
struct DocumentUpdate {
    /// A new title.
    title: Option<String>,
    /// A new revision count.
    revisions: Option<u32>,
}

impl State for Document {
    type Update = DocumentUpdate;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![
            ChannelSpec::new("title")?,
            ChannelSpec::new("revisions")?,
        ])
    }

    fn apply_step(&mut self, updates: Vec<(TaskPath, DocumentUpdate)>) -> Result<(), ReduceError> {
        let titles = updates
            .iter()
            .filter_map(|(p, u)| u.title.clone().map(|t| (p, t)));
        reducer::replace("title", &mut self.title, titles)?;
        let revisions = updates
            .iter()
            .filter_map(|(p, u)| u.revisions.map(|r| (p, r)));
        reducer::replace("revisions", &mut self.revisions, revisions)
    }
}

#[tokio::test]
async fn a_two_step_graph_over_a_two_field_state() {
    /// Drafts the document: sets its title and first revision.
    async fn draft(
        _s: Arc<Document>,
        _ctx: NodeContext<()>,
    ) -> Result<Option<DocumentUpdate>, BoxError> {
        Ok(Some(DocumentUpdate {
            title: Some("Draft".to_owned()),
            revisions: Some(1),
        }))
    }

    let mut g = GraphBuilder::<Document>::new();
    let draft = g.add_node("draft", node::from_fn(draft)).unwrap();
    let revise = g
        .add_node(
            "revise",
            node::from_fn(|s: Arc<Document>, _ctx| async move {
                Ok(Some(DocumentUpdate {
                    title: None,
                    revisions: Some(s.revisions + 1),
                }))
            }),
        )
        .unwrap();
    g.add_edge(START, draft).unwrap();
    g.add_edge(draft, revise).unwrap();
    g.add_edge(revise, END).unwrap();

    let out = g
        .compile()
        .unwrap()
        .invoke(DocumentUpdate::default(), RunConfig::default())
        .await
        .unwrap();
    assert_eq!(
        out,
        Document {
            title: "Draft".to_owned(),
            revisions: 2
        }
    );
}

/// A run context carrying a dependency nodes read.
#[derive(Debug)]
struct Pricing {
    /// The tax rate, in percent.
    tax_percent: i64,
}

#[tokio::test]
async fn nodes_read_a_typed_run_context() {
    let mut g = GraphBuilder::<Counter, Pricing>::new();
    let tax = g
        .add_node(
            "tax",
            node::from_fn(|s: Arc<Counter>, ctx: NodeContext<Pricing>| async move {
                let total = s.count + s.count * ctx.context().tax_percent / 100;
                Ok(Some(CounterUpdate { count: Some(total) }))
            }),
        )
        .unwrap();
    g.add_edge(START, tax).unwrap();
    g.add_edge(tax, END).unwrap();

    let config = RunConfig::new(Pricing { tax_percent: 20 });
    let out = g
        .compile()
        .unwrap()
        .invoke(CounterUpdate { count: Some(100) }, config)
        .await
        .unwrap();
    assert_eq!(out.count, 120);
}

#[tokio::test]
async fn a_cycle_runs_until_the_recursion_limit() {
    let mut g = GraphBuilder::<Counter>::new();
    let ping = g
        .add_node(
            "ping",
            node::from_fn(|s: Arc<Counter>, _ctx| async move {
                Ok(Some(CounterUpdate {
                    count: Some(s.count + 1),
                }))
            }),
        )
        .unwrap();
    let reply = g
        .add_node("reply", node::from_fn(|_s, _ctx| async move { Ok(()) }))
        .unwrap();
    g.add_edge(START, ping).unwrap();
    g.add_edge(ping, reply).unwrap();
    g.add_edge(reply, ping).unwrap();

    let config = RunConfig::default().with_recursion_limit(std::num::NonZeroU32::new(10).unwrap());
    let error = g
        .compile()
        .unwrap()
        .invoke(CounterUpdate::default(), config)
        .await
        .unwrap_err();
    assert!(matches!(error, RunError::RecursionLimit { .. }));
}

#[tokio::test]
async fn nodes_capture_values_by_cloning_into_an_async_block() {
    // A capturing `async` closure does not implement `Fn`, so the closure
    // clones its non-`Copy` capture into each call's `async move` block.
    let prefix = Arc::new(String::from("count: "));
    let mut g = GraphBuilder::<Document>::new();
    let label = g
        .add_node(
            "label",
            node::from_fn(move |s: Arc<Document>, _ctx| {
                let prefix = Arc::clone(&prefix);
                async move {
                    Ok(Some(DocumentUpdate {
                        title: Some(format!("{prefix}{}", s.revisions)),
                        revisions: None,
                    }))
                }
            }),
        )
        .unwrap();
    g.add_edge(START, label).unwrap();
    g.add_edge(label, END).unwrap();

    let input = DocumentUpdate {
        title: None,
        revisions: Some(3),
    };
    let out = g
        .compile()
        .unwrap()
        .invoke(input, RunConfig::default())
        .await
        .unwrap();
    assert_eq!(out.title, "count: 3");
}

/// A fan-out: two branches run in the same superstep, each writing its own
/// channel, and both updates are applied together.
#[tokio::test]
async fn two_branches_write_different_channels_in_one_step() {
    let mut g = GraphBuilder::<Document>::new();
    let title = g
        .add_node(
            "title",
            node::from_fn(|_s, _ctx| async move {
                Ok(Some(DocumentUpdate {
                    title: Some("Report".to_owned()),
                    revisions: None,
                }))
            }),
        )
        .unwrap();
    let revise = g
        .add_node(
            "revise",
            node::from_fn(|s: Arc<Document>, _ctx| async move {
                Ok(Some(DocumentUpdate {
                    title: None,
                    revisions: Some(s.revisions + 1),
                }))
            }),
        )
        .unwrap();
    g.add_edge(START, title).unwrap();
    g.add_edge(START, revise).unwrap();
    g.add_edge(title, END).unwrap();
    g.add_edge(revise, END).unwrap();
    let graph = g.compile().unwrap();

    let document = graph
        .invoke(DocumentUpdate::default(), RunConfig::default())
        .await
        .unwrap();
    assert_eq!(
        document,
        Document {
            title: "Report".to_owned(),
            revisions: 1,
        }
    );
}

/// A research state that parallel branches write, through the three merging
/// reducers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Research {
    /// What the branches found, appended in task-path order.
    findings: Vec<String>,
    /// What the branches cost, summed.
    cost: u64,
    /// The research's tags, changed by tag operations.
    tags: BTreeSet<String>,
}

/// An operation on [`Research::tags`].
#[derive(Debug, Clone, Copy)]
enum Tag {
    /// Adds a tag.
    Add(&'static str),
    /// Removes a tag, which must be present.
    Remove(&'static str),
}

/// An update to [`Research`]; empty fields leave their channel unwritten.
#[derive(Debug, Default)]
struct ResearchUpdate {
    /// Findings to append.
    findings: Vec<String>,
    /// A cost to add.
    cost: Option<u64>,
    /// Tag operations to apply.
    tags: Vec<Tag>,
}

/// Applies one tag operation; removing a missing tag is an error.
fn apply_tag(tags: &mut BTreeSet<String>, op: Tag) -> Result<(), BoxError> {
    match op {
        Tag::Add(tag) => {
            tags.insert(tag.to_owned());
            Ok(())
        }
        Tag::Remove(tag) if tags.remove(tag) => Ok(()),
        Tag::Remove(tag) => Err(format!("no tag `{tag}` to remove").into()),
    }
}

impl State for Research {
    type Update = ResearchUpdate;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![
            ChannelSpec::new("findings")?,
            ChannelSpec::new("cost")?,
            ChannelSpec::new("tags")?,
        ])
    }

    fn apply_step(
        &mut self,
        mut updates: Vec<(TaskPath, ResearchUpdate)>,
    ) -> Result<(), ReduceError> {
        let findings = updates
            .iter_mut()
            .map(|(p, u)| (&*p, mem::take(&mut u.findings)));
        reducer::append(&mut self.findings, findings);

        let costs = updates.iter().filter_map(|(p, u)| u.cost.map(|c| (p, c)));
        reducer::combine(&mut self.cost, costs, |total, c| *total += c);

        let tags = updates.iter_mut().flat_map(|(p, u)| {
            let p: &TaskPath = p;
            mem::take(&mut u.tags).into_iter().map(move |op| (p, op))
        });
        reducer::fold("tags", &mut self.tags, tags, apply_tag)
    }
}

/// Two branches write the same three channels in one step, and the reducers
/// merge their writes in task-path order instead of conflicting: `alpha`'s
/// before `beta`'s, so `beta` can remove the tag `alpha` added.
#[tokio::test]
async fn a_fan_out_merges_its_branches_with_reducers() {
    let mut g = GraphBuilder::<Research>::new();
    let beta = g
        .add_node(
            "beta",
            node::from_fn(|_s, _ctx| async move {
                Ok(Some(ResearchUpdate {
                    findings: vec!["beta found B".to_owned()],
                    cost: Some(5),
                    tags: vec![Tag::Remove("unreviewed"), Tag::Add("reviewed")],
                }))
            }),
        )
        .unwrap();
    let alpha = g
        .add_node(
            "alpha",
            node::from_fn(|_s, _ctx| async move {
                Ok(Some(ResearchUpdate {
                    findings: vec!["alpha found A".to_owned()],
                    cost: Some(3),
                    tags: vec![Tag::Add("unreviewed")],
                }))
            }),
        )
        .unwrap();
    for branch in [beta, alpha] {
        g.add_edge(START, branch).unwrap();
        g.add_edge(branch, END).unwrap();
    }
    let graph = g.compile().unwrap();

    let research = graph
        .invoke(ResearchUpdate::default(), RunConfig::default())
        .await
        .unwrap();
    assert_eq!(
        research,
        Research {
            findings: vec!["alpha found A".to_owned(), "beta found B".to_owned()],
            cost: 8,
            tags: BTreeSet::from(["reviewed".to_owned()]),
        }
    );
}

/// A report that two researchers' findings feed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Report {
    /// The researchers' findings, appended in task-path order.
    findings: Vec<String>,
    /// The drafts the writer writes from them, one per run, appended.
    drafts: Vec<String>,
}

/// An update to [`Report`]; empty fields leave their channel unwritten.
#[derive(Debug, Default)]
struct ReportUpdate {
    /// Findings to append.
    findings: Vec<String>,
    /// A new draft.
    draft: Option<String>,
}

impl State for Report {
    type Update = ReportUpdate;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![
            ChannelSpec::new("findings")?,
            ChannelSpec::new("drafts")?,
        ])
    }

    fn apply_step(
        &mut self,
        mut updates: Vec<(TaskPath, ReportUpdate)>,
    ) -> Result<(), ReduceError> {
        let findings = updates
            .iter_mut()
            .map(|(p, u)| (&*p, mem::take(&mut u.findings)));
        reducer::append(&mut self.findings, findings);
        let drafts = updates.iter_mut().map(|(p, u)| (&*p, u.draft.take()));
        reducer::append(&mut self.drafts, drafts);
        Ok(())
    }
}

/// Two researchers run in parallel, and the writer, joined to both, runs
/// once, after the slower of them, seeing both findings: one draft is
/// written, not one per researcher.
#[tokio::test]
async fn two_researchers_joined_by_a_writer() {
    let mut g = GraphBuilder::<Report>::new();
    let alpha = g
        .add_node(
            "alpha",
            node::from_fn(|_s, _ctx| async move {
                Ok(Some(ReportUpdate {
                    findings: vec!["alpha found A".to_owned()],
                    draft: None,
                }))
            }),
        )
        .unwrap();
    let gather = g
        .add_node("gather", node::from_fn(|_s, _ctx| async move { Ok(()) }))
        .unwrap();
    let beta = g
        .add_node(
            "beta",
            node::from_fn(|_s, _ctx| async move {
                Ok(Some(ReportUpdate {
                    findings: vec!["beta found B".to_owned()],
                    draft: None,
                }))
            }),
        )
        .unwrap();
    let writer = g
        .add_node(
            "writer",
            node::from_fn(|s: Arc<Report>, _ctx| async move {
                Ok(Some(ReportUpdate {
                    findings: Vec::new(),
                    draft: Some(s.findings.join("; ")),
                }))
            }),
        )
        .unwrap();
    // `beta` is one step slower: it gathers first.
    g.add_edge(START, alpha).unwrap();
    g.add_edge(START, gather).unwrap();
    g.add_edge(gather, beta).unwrap();
    g.add_edge([alpha, beta], writer).unwrap();
    g.add_edge(writer, END).unwrap();
    let graph = g.compile().unwrap();

    let report = graph
        .invoke(ReportUpdate::default(), RunConfig::default())
        .await
        .unwrap();
    assert_eq!(report.drafts, ["alpha found A; beta found B"]);
}
