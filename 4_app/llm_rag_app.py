!ldconfig -v

import os
import gradio

from pymilvus import Collection
import torch
from utils.model_llm_utils import load_models
import utils.model_embedding_utils as model_embedding
import utils.vector_db_utils as vector_db

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llm_model, tokenizer=load_models(device)
def main():
    # Gradio 인터페이스 설정
    iface = gradio.Interface(
        fn=answer_questions, 
        inputs=gradio.Textbox(lines=2, placeholder="여기에 질문을 입력하세요..."), 
        outputs=[gradio.Textbox(label="Llama로 생성된 답변"),
                 gradio.Textbox(label="벡터 DB에서 찾은 문맥을 바탕으로 생성된 답변")],
         examples=["ML Runtimes이 뭔가요?",
                    "어떤 유형의 사용자가 CML을 사용하나요?",
                    "데이터 과학자는 어떤 방식으로 CML을 사용하나요?",
                    "iceberg tables란?"],
        title="질의응답 시스템",
        description="첫 번째 출력은 Llama로 생성된 답변이고, 두 번째 출력은 벡터 DB에서 찾은 문맥을 바탕으로 생성된 답변입니다."
    )

    # Launch gradio app
    print("Launching gradio app")
    iface.launch(share=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")


# Helper function for generating responses for the QA app
def answer_questions(question, vector_db_collection_name = 'cloudera_ml_docs'):
    vector_db_collection = Collection(name=vector_db_collection_name)
    vector_db_collection.load()

    answer_with_context = answer_question_with_context(vector_db_collection, question, device)
    answer_without_context = answer_question_without_context(question, device)
    return answer_without_context, answer_with_context


def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()
      
def get_nearest_chunk_from_vectordb(vector_db_collection, question):
    # Generate embedding for user question
    question_embedding =  model_embedding.get_embeddings(question)
    
    # Define search attributes for Milvus vector DB
    vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # Execute search and get nearest vector, outputting the relativefilepath
    nearest_vectors = vector_db_collection.search(
        data=[question_embedding], # The data you are querying on
        anns_field="embedding", # Column in collection to search on
        param=vector_db_search_params,
        limit=1, # limit results to 1
        expr=None, 
        output_fields=['relativefilepath'], # The fields you want to retrieve from the search result.
        consistency_level="Strong"
    )
    
    # Print the file path of the kb chunk
    print(nearest_vectors[0].ids[0])
    
    # Return text of the nearest knowledgebase chunk
    return load_context_chunk_from_data(nearest_vectors[0].ids[0])
  
  
def gen(x, model, tokenizer, device):
    prompt = (
        f"아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{x}\n\n### 응답:"
    )
    len_prompt = len(prompt)
    gened = model.generate(
        **tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device),
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
        top_k=20,
        top_p=0.92,
        no_repeat_ngram_size=3,
        eos_token_id=2,
        repetition_penalty=1.2,
        num_beams=3
    )
    return tokenizer.decode(gened[0])[len_prompt:]

def answer_question_with_context(vector_db_collection, question, device):
    context_chunk = get_nearest_chunk_from_vectordb(vector_db_collection, question)
    return gen("문맥: " +context_chunk + "질문 : " + question 
            , model=llm_model
            , tokenizer=tokenizer
            , device=device)

def answer_question_without_context(question, device):
    return gen(question
                , model=llm_model
                , tokenizer=tokenizer
                , device=device)

if __name__ == "__main__":
    main()
