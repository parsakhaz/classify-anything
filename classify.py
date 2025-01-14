import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, pipeline
import os, gc, glob
from typing import List
from huggingface_hub import login, whoami

torch.cuda.empty_cache()
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_authentication(token: str = None):
    """Setup HuggingFace authentication either via web UI or token."""
    print("\nChecking HuggingFace authentication...")
    print("This is required to access the LLaMA model for narrative generation.")
    print("Note: You must have been granted access to Meta's LLaMA model.")
    print("      If you haven't requested access yet, visit:")
    print("      https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    
    try:
        if token:
            print("Using provided HuggingFace token...")
            login(token=token)
        else:
            try:
                # Try to get existing login
                user_info = whoami()
                print(f"Found existing login as: {user_info['name']} ({user_info['email']})")
                return True  # Successfully authenticated
            except Exception:
                print("\nNo existing login found. You can:")
                print("1. Pre-authenticate using the command line:")
                print("   $ huggingface-cli login")
                print("2. Run this script with a token:")
                print("   $ python classify.py --token YOUR_TOKEN")
                print("\nPlease authenticate using one of these methods and try again.")
                return False
        
        # Verify authentication
        user_info = whoami()
        print(f"\nSuccessfully authenticated as: {user_info['name']} ({user_info['email']})")
        return True
        
    except Exception as e:
        print("\nAuthentication Error:")
        print("Make sure you have:")
        print("1. A HuggingFace account")
        print("2. Requested and been granted access to Meta's LLaMA model")
        print("   Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        print("   Note: The approval process may take several days")
        print("3. Either:")
        print("   - Run 'huggingface-cli login' in your terminal")
        print("   - Provide a valid token with --token")
        
        if "Cannot access gated repo" in str(e) or "awaiting a review" in str(e):
            print("\nError: You don't have access to the LLaMA model yet.")
            print("Please request access and wait for approval before using this script.")
        else:
            print("\nError details:", str(e))
        return False

def load_moondream():
    """Load Moondream model."""
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09", 
        trust_remote_code=True,
        device_map={"": "cuda"}
    )

def load_llama():
    """Load LLaMA model."""
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

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
    import argparse
    parser = argparse.ArgumentParser(description='Classify images using Moondream and LLaMA')
    parser.add_argument('--token', type=str, help='Optional: HuggingFace token for authentication')
    args = parser.parse_args()

    # Setup authentication before loading models
    if not setup_authentication(args.token):
        print("\nAuthentication failed. Please fix authentication issues before continuing.")
        return

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