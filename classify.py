import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, pipeline
import os, gc, glob
from typing import List

torch.cuda.empty_cache()
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_moondream():
    """Load Moondream model."""
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-01-09", 
        trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()

def load_llama():
    """Load LLaMA model."""
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", 
        torch_dtype=torch.bfloat16, device_map="auto")

def create_questions(llm, things_to_classify: List[str]) -> List[str]:
    """Create all questions at once."""
    print("\nFormulating questions using LLaMA...")
    questions = []
    for thing in things_to_classify:
        print("Input prompt:", thing)
        response = llm([{
            "role": "system", 
            "content": """You format inputs into clear questions for an image model. Output ONLY the question, with no additional text or explanations. For example:
Input: "grass color"
Output: "What is the color of the grass?"

Keep questions focused and direct. Do not include any other text besides the question itself."""
        }, {
            "role": "user",
            "content": f"i am trying to understand the following thing about an image: {thing}. respond with a question only."
        }], max_new_tokens=256, do_sample=True, temperature=0.7, pad_token_id=2)
        question = response[0]["generated_text"][-1]["content"].strip() + " Answer concisely, in a few words."
        questions.append(question)
        print(f"Generated question for '{thing}': {question}")
    return questions

def classify_batch(file_path: str, things_to_classify: List[str]):
    """Classify multiple aspects of an image."""
    if file_path.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        raise ValueError("File type not supported. Only images are supported.")
    
    try:
        print("\nLoading LLaMA to generate questions...")
        llm = load_llama()
        questions = create_questions(llm, things_to_classify)
        
        print("\nUnloading LLaMA and loading Moondream...")
        del llm
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        moondream = load_moondream()
        pil_image = Image.open(file_path)

        results = []
        for thing, question in zip(things_to_classify, questions):
            print(f"\nProcessing aspect: {thing}")
            print(f"Question: {question}")
            answer = moondream.query(pil_image, question)["answer"]
            results.append((thing, answer))
            print(f"Answer: {answer}")
        return results
            
    finally:
        if 'moondream' in locals(): del moondream
        if 'llm' in locals(): del llm
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

def main():
    inputs_dir = "inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    
    input_files = []
    for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        input_files.extend(glob.glob(os.path.join(inputs_dir, f"*.{ext}")))
    
    if not input_files:
        print("No supported files found in 'inputs' directory.")
        print("Supported formats: jpg, jpeg, png, bmp, tiff")
        return
    
    things_to_classify = ["grass color", "time of day", "number of people, if any", 
                         "weather conditions", "main activity"]
    
    for file_path in input_files:
        print("\n" + "="*70)
        print(f"Processing: {os.path.basename(file_path)}")
        print("="*70)
        try:
            results = classify_batch(file_path, things_to_classify)
            print("\nFinal Classification Results:")
            print("-"*30)
            for thing, result in results:
                print(f"Aspect: {thing}")
                print(f"Result: {result}")
                print("-"*30)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()