//! Happy-path examples of the public API, written to be read.

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
