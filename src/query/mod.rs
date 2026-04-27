pub mod bm25;
pub mod executor;
pub mod keywords;
pub mod parser;
pub mod planner;
pub mod rag;
pub mod rag_config;
pub mod rag_retrieval;
pub mod reranker;
pub mod session;
pub mod token_budget;

// Fuzzy Sprint Phase 3 — TensaQL tail clauses. Cites: [klement2000]
// [yager1988owa] [grabisch1996choquet].
#[cfg(test)]
mod executor_fuzzy_tests;
#[cfg(test)]
mod parser_fuzzy_tests;
#[cfg(test)]
mod planner_fuzzy_tests;
