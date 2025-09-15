import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_community.retrievers import WikipediaRetriever
import os
from dotenv import load_dotenv

os.environ["GROQ_API_KEY"] = "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"
load_dotenv()

st.title("🧠 Claim Validator")

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7, max_tokens=1024)

raw_claim = st.text_input("Enter a claim (e.g., moon, shawarma):")

if st.button("Validate Claim") and raw_claim.strip():
    topic_retrieve = PromptTemplate.from_template(
        "Extract the most specific noun or concept from this claim: {input}. Return only one word."
    )
    topic_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    topic_chain = LLMChain(llm=topic_llm, prompt=topic_retrieve)
    topic_name = topic_chain.run(input=raw_claim)
    st.write("🔍 Topic extracted:", topic_name)

    retriever = WikipediaRetriever(lang="en", load_max_docs=1)
    docs = retriever.get_relevant_documents(topic_name)
    context = docs[0].page_content[:1000] if docs else "No Wikipedia context found."
    st.write("📚 Wikipedia Context:", context[:500])

    validate_prompt = PromptTemplate.from_template(
        "Fact check this claim: {claim}. according to this context: {context}. Respond with 'True' or 'False' and a brief explanation."
    )
    validate_chain = LLMChain(llm=llm, prompt=validate_prompt)

    with st.spinner("Validating claim..."):
        validation = validate_chain.run({"claim": raw_claim, "context": context})
    st.success("✅ Validation Result:")
    st.write(validation)
