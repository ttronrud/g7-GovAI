QUERY_INSTRUCTIONS = "You are a helpful government assistant." 
QUERY_PROMPT = """Read the following proposal and come up with a very short query that can be used to ask another model what applicable regulations should be looked up based on the combination of proposal details, industry, and location.
# Start of Proposal
{proposal_txt}
# End of Proposal
"""

#RERANK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
RERANK_INSTRUCTION = 'Given a search query, retrieve relevant documents that answer the query'
RERANK_PROMPT = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

SUMMARY_PROMPT = """Your job is to summarize the relevant aspects of a specific regulation with respect to a proposal. The proposal is provided first, followed by the regulation text.

# Start of Proposal
{prop_txt}
# End of Proposal


# Start of Regulation
{reg_title}
{reg_txt}
# End of Regulation

Extract the potentially relevant aspects of the regulation based on the proposal and briefly summarize them as compactly as possible. Do not make a determination on whether the proposal meets the regulations or not. Include whether the location(s) specified in the proposal fall under the jurisdiction of the regulation. Regulations can exist to exempt locations from other regulations. Include whether the specific industry targeted by the regulation matches the proposal. Reason and respond succinctly.
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
