from agent.prompts.request_evidence_prompt import REQUEST_EVIDENCE_PROMPT
from models.llm import get_mapping_llm
from models.structured_outputs import RequestEvidenceOutput


def get_request_evidence_chain(include_raw: bool = False):
    llm = get_mapping_llm().with_structured_output(
        RequestEvidenceOutput,
        include_raw=include_raw,
    )

    return REQUEST_EVIDENCE_PROMPT | llm
