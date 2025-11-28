QUERY_INSTRUCTIONS = "You are an expert summarizer of reports and applications." 
QUERY_PROMPT = """Read the following application and come up 
with a sentence that can be used to ask another model what applicable health and safety regulations should be looked up based on the type of company in the application. \n\n
{proposal_txt}

"""

RERANK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
RERANK_PROMPT = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

SUMMARY_PROMPT = """Your job is to summarize the relevant aspects of a regulation with respect to a proposal. The proposal is provided first, followed by the regulation text.

# Start of Proposal
{prop_txt}
# End of Proposal


# Start of Regulation
{reg_txt}
# End of Regulation

Extract the potentially relevant aspects of the regulation based on the proposal and briefly summarize them as compactly as possible. Do not make a determination on whether the proposal meets the regulations or not. Include whether the location(s) specified in the proposal fall under the jurisdiction of the regulation. Reason and respond succinctly.
"""
OUTPUT_JSON_FORMAT = "{'applicable':[true/false/uncertain], 'violation':[true/false/uncertain], 'notes':[additional information and context regarding applicability and violation status]}"

OUTPUT_PROMPT = """Your job is to determine whether a regulation is applicable to a proposal. Use a summary of the most likely relevant aspects of the regulations and their scopes to make your decision.

# Start of Proposal
{reg_title}
{prop_txt}
# End of Proposal

# Start of Regulation Summary
{reg_summ_txt}
# End of Regulation Summary

Does this proposal violate the regulation? Reason and respond succinctly. Respond in JSON format of the form:
{json_form}
"""
