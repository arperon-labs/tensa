//! GraphML export for narrative data (Sprint P3.6 — F-AP4).
//!
//! Generates GraphML XML with entities and situations as nodes,
//! participations and causal links as edges.

use crate::error::Result;
use crate::export::{entity_display_name, NarrativeExport};

/// Escape XML special characters. Returns borrowed input when no escaping needed.
fn xml_escape(s: &str) -> std::borrow::Cow<'_, str> {
    if s.bytes()
        .all(|b| b != b'&' && b != b'<' && b != b'>' && b != b'"' && b != b'\'')
    {
        std::borrow::Cow::Borrowed(s)
    } else {
        std::borrow::Cow::Owned(
            s.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
                .replace('\'', "&apos;"),
        )
    }
}

/// Export narrative data as GraphML XML string.
pub fn export_graphml(data: &NarrativeExport) -> Result<String> {
    let estimated_size = (data.entities.len() + data.situations.len()) * 200
        + data.participations.len() * 100
        + 2048;
    let mut xml = String::with_capacity(estimated_size);

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/graphml\"\n");
    xml.push_str("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/graphml http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");
    xml.push_str("  <key id=\"d0\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d1\" for=\"node\" attr.name=\"node_type\" attr.type=\"string\"/>\n");
    xml.push_str(
        "  <key id=\"d2\" for=\"node\" attr.name=\"entity_type\" attr.type=\"string\"/>\n",
    );
    xml.push_str(
        "  <key id=\"d3\" for=\"node\" attr.name=\"narrative_level\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"d4\" for=\"node\" attr.name=\"confidence\" attr.type=\"double\"/>\n");
    xml.push_str("  <key id=\"d5\" for=\"edge\" attr.name=\"edge_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d6\" for=\"edge\" attr.name=\"role\" attr.type=\"string\"/>\n");
    xml.push_str(
        "  <key id=\"d7\" for=\"edge\" attr.name=\"causal_type\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"d8\" for=\"edge\" attr.name=\"strength\" attr.type=\"double\"/>\n");

    xml.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    for entity in &data.entities {
        let name = entity_display_name(entity, "unnamed");
        let etype = format!("{:?}", entity.entity_type);

        xml.push_str(&format!("    <node id=\"{}\">\n", entity.id));
        xml.push_str(&format!(
            "      <data key=\"d0\">{}</data>\n",
            xml_escape(name)
        ));
        xml.push_str("      <data key=\"d1\">entity</data>\n");
        xml.push_str(&format!(
            "      <data key=\"d2\">{}</data>\n",
            xml_escape(&etype)
        ));
        xml.push_str(&format!(
            "      <data key=\"d4\">{}</data>\n",
            entity.confidence
        ));
        xml.push_str("    </node>\n");
    }

    for sit in &data.situations {
        let desc = sit
            .raw_content
            .first()
            .map(|c| c.content.as_str())
            .unwrap_or("");
        let level = format!("{:?}", sit.narrative_level);

        xml.push_str(&format!("    <node id=\"{}\">\n", sit.id));
        xml.push_str(&format!(
            "      <data key=\"d0\">{}</data>\n",
            xml_escape(desc)
        ));
        xml.push_str("      <data key=\"d1\">situation</data>\n");
        xml.push_str(&format!(
            "      <data key=\"d3\">{}</data>\n",
            xml_escape(&level)
        ));
        xml.push_str(&format!(
            "      <data key=\"d4\">{}</data>\n",
            sit.confidence
        ));
        xml.push_str("    </node>\n");
    }

    for (i, part) in data.participations.iter().enumerate() {
        let role = format!("{:?}", part.role);
        xml.push_str(&format!(
            "    <edge id=\"p{i}\" source=\"{}\" target=\"{}\">\n",
            part.entity_id, part.situation_id
        ));
        xml.push_str("      <data key=\"d5\">participation</data>\n");
        xml.push_str(&format!(
            "      <data key=\"d6\">{}</data>\n",
            xml_escape(&role)
        ));
        xml.push_str("    </edge>\n");
    }

    for (i, link) in data.causal_links.iter().enumerate() {
        let ctype = format!("{:?}", link.causal_type);
        xml.push_str(&format!(
            "    <edge id=\"c{i}\" source=\"{}\" target=\"{}\">\n",
            link.from_situation, link.to_situation
        ));
        xml.push_str("      <data key=\"d5\">causal</data>\n");
        xml.push_str(&format!(
            "      <data key=\"d7\">{}</data>\n",
            xml_escape(&ctype)
        ));
        xml.push_str(&format!(
            "      <data key=\"d8\">{}</data>\n",
            link.strength
        ));
        xml.push_str("    </edge>\n");
    }

    xml.push_str("  </graph>\n");
    xml.push_str("</graphml>\n");

    Ok(xml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::NarrativeExport;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_entity(name: &str) -> Entity {
        let now = Utc::now();
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test".into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(desc: &str) -> Situation {
        let now = Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(desc)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("test".into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn test_graphml_valid_structure() {
        let data = NarrativeExport {
            narrative_id: "test".into(),
            entities: vec![make_entity("Alice")],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
        };
        let xml = export_graphml(&data).unwrap();
        assert!(xml.contains("<?xml version="));
        assert!(xml.contains("<graphml"));
        assert!(xml.contains("graphdrawing.org"));
        assert!(xml.contains("<graph"));
        assert!(xml.contains("</graph>"));
        assert!(xml.contains("</graphml>"));
        assert!(xml.contains("<key id="));
    }

    #[test]
    fn test_graphml_nodes_count() {
        let e1 = make_entity("Alice");
        let e2 = make_entity("Bob");
        let s1 = make_situation("Meeting");
        let data = NarrativeExport {
            narrative_id: "test".into(),
            entities: vec![e1, e2],
            situations: vec![s1],
            participations: vec![],
            causal_links: vec![],
        };
        let xml = export_graphml(&data).unwrap();
        let node_count = xml.matches("<node id=").count();
        assert_eq!(node_count, 3); // 2 entities + 1 situation
    }

    #[test]
    fn test_graphml_participation_edges() {
        let e = make_entity("Alice");
        let s = make_situation("Meeting");
        let eid = e.id;
        let sid = s.id;
        let data = NarrativeExport {
            narrative_id: "test".into(),
            entities: vec![e],
            situations: vec![s],
            participations: vec![Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: Some("speaks".into()),
                payoff: None,
                seq: 0,
            }],
            causal_links: vec![],
        };
        let xml = export_graphml(&data).unwrap();
        assert!(xml.contains("<edge id=\"p0\""));
        assert!(xml.contains("participation"));
        assert!(xml.contains("Protagonist"));
    }

    #[test]
    fn test_graphml_causal_edges() {
        let s1 = make_situation("Cause");
        let s2 = make_situation("Effect");
        let s1id = s1.id;
        let s2id = s2.id;
        let data = NarrativeExport {
            narrative_id: "test".into(),
            entities: vec![],
            situations: vec![s1, s2],
            participations: vec![],
            causal_links: vec![CausalLink {
                from_situation: s1id,
                to_situation: s2id,
                mechanism: Some("direct".into()),
                strength: 0.8,
                causal_type: CausalType::Necessary,
                maturity: MaturityLevel::Candidate,
            }],
        };
        let xml = export_graphml(&data).unwrap();
        assert!(xml.contains("<edge id=\"c0\""));
        assert!(xml.contains("causal"));
        assert!(xml.contains("Necessary"));
        assert!(xml.contains("0.8"));
    }
}
