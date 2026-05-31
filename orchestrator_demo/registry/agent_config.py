"""Default specialist agent registry config.

This module is intentionally plain Python so deployments can reload a changed
descriptor list without restarting the orchestrator process.
"""

from orchestrator_demo.contracts import AgentDescriptor


AVAILABLE_AGENTS = [
    AgentDescriptor(
        agent_id="industry_research",
        display_name="Industry Research Agent",
        capabilities=["industry trends", "market risks", "sector outlook"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Give me a quick overview of the manufacturing industry.",
            "What are key risks in retail trade this quarter?",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="web_search",
        display_name="Web Search Agent",
        capabilities=["public company research", "market events", "source summaries"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Research recent public information about this prospect.",
            "Summarize public market events for ABC Manufacturing.",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="internal_knowledge",
        display_name="Internal Knowledge Agent",
        capabilities=["crm notes", "policy snippets", "relationship records"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Summarize the internal notes for ABC Manufacturing.",
            "Find open follow-ups from the latest relationship notes.",
        ],
        execution_mode="local_a2a_compatible",
    ),
    AgentDescriptor(
        agent_id="credit_risk",
        display_name="Credit Risk Agent",
        capabilities=["risk themes", "covenants", "repayment indicators"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Flag credit risks for this customer.",
            "What repayment concerns should I review before the meeting?",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="relationship_summary",
        display_name="Relationship Summary Agent",
        capabilities=["relationship history", "contacts", "prior meetings"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Summarize the relationship history for ABC Manufacturing.",
            "Who are the key contacts for this customer?",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="product_opportunity",
        display_name="Product Opportunity Agent",
        capabilities=[
            "deposit opportunities",
            "lending opportunities",
            "treasury services",
        ],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "What product opportunities should I consider for a cafe business?",
            "Identify treasury opportunities for this customer.",
        ],
        execution_mode="local_a2a_compatible",
    ),
    AgentDescriptor(
        agent_id="compliance_policy",
        display_name="Compliance / Policy Agent",
        capabilities=["policy checks", "regulated-output review", "caveat guidance"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Check whether this response makes unsupported regulated claims.",
            "Add policy caveats for a credit-sensitive customer discussion.",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="data_quality",
        display_name="Data Quality Agent",
        capabilities=["missing data", "stale context", "evidence confidence"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Highlight missing information in this customer context.",
            "What evidence is weak or stale in this briefing?",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="meeting_prep",
        display_name="Meeting Prep Agent",
        capabilities=["meeting briefs", "talking points", "follow-up questions"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Prepare me for tomorrow's meeting with ABC Manufacturing.",
            "Create talking points for my customer meeting.",
        ],
        execution_mode="local_llm",
    ),
    AgentDescriptor(
        agent_id="synthesis",
        display_name="Synthesis Agent",
        capabilities=["multi-agent synthesis", "briefing assembly", "final answer"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Combine specialist findings into an RM-ready answer.",
            "Create a final briefing from the completed plan outputs.",
        ],
        execution_mode="local_llm",
    ),
]


__all__ = ["AVAILABLE_AGENTS"]
