import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, pipeline
import os, gc, glob
from typing import List
from huggingface_hub import login, whoami
import requests
import subprocess
import platform
import time
import json
import tempfile
import shutil
from datetime import datetime

torch.cuda.empty_cache()
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class OllamaWrapper:
    def __call__(self, messages, **kwargs):
        prompt = messages[1]["content"]  # Get user content from messages
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }
        )
        return [{"generated_text": [{
            "role": "assistant",
            "content": response.json()["response"]
        }]}]

def get_os_type():
    """Determine the OS type."""
    system = platform.system().lower()
    if system == "darwin": return "mac"
    elif system == "windows": return "windows"
    else: return "linux"

def install_ollama():
    """Install Ollama based on the OS."""
    os_type = get_os_type()
    print(f"\nDetected OS: {os_type.capitalize()}")
    print("Attempting to install Ollama...")

    try:
        if os_type == "mac":
            try:
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
            except:
                print("Homebrew not found. Installing Homebrew...")
                subprocess.run(['/bin/bash', '-c', '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'])
            print("Installing Ollama via Homebrew...")
            subprocess.run(["brew", "install", "ollama"])

        elif os_type == "linux":
            print("Installing Ollama via curl...")
            install_cmd = 'curl https://ollama.ai/install.sh | sh'
            subprocess.run(install_cmd, shell=True, check=True)

        elif os_type == "windows":
            print("Downloading Ollama installer...")
            temp_dir = tempfile.mkdtemp()
            installer_path = os.path.join(temp_dir, "ollama-installer.msi")
            
            response = requests.get("https://ollama.ai/download/windows", stream=True)
            with open(installer_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            print("Installing Ollama...")
            subprocess.run(['msiexec', '/i', installer_path, '/quiet'], check=True)
            shutil.rmtree(temp_dir)

        print("Ollama installation completed!")
        
    except Exception as e:
        print(f"\nError installing Ollama: {str(e)}")
        print("\nPlease install Ollama manually:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install the appropriate version for your OS")
        print("3. Run the script again after installation")
        raise Exception("Ollama installation failed")

def start_ollama_service():
    """Start the Ollama service based on OS."""
    os_type = get_os_type()
    
    try:
        if os_type == "windows":
            try:
                requests.get("http://localhost:11434/api/tags")
                return
            except:
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux and Mac
            try:
                requests.get("http://localhost:11434/api/tags")
                return
            except:
                subprocess.Popen(['ollama', 'serve'])
        
        print("Starting Ollama service...")
        max_retries = 10
        for i in range(max_retries):
            try:
                requests.get("http://localhost:11434/api/tags")
                print("Ollama service started successfully!")
                return
            except:
                if i < max_retries - 1:
                    print(f"Waiting for service to start... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    raise Exception("Service failed to start")
                
    except Exception as e:
        print(f"\nError starting Ollama service: {str(e)}")
        print("Please start Ollama manually and try again")
        raise

def pull_llama_model():
    """Pull the LLaMA model in Ollama."""
    print("\nPulling LLaMA model...")
    try:
        subprocess.run(['ollama', 'pull', 'llama3.2:1b'], check=True)
        print("LLaMA model pulled successfully!")
    except Exception as e:
        print(f"\nError pulling LLaMA model: {str(e)}")
        raise

def setup_ollama():
    """Setup and verify Ollama LLaMA."""
    print("\nChecking Ollama setup...")
    
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except:
        install_ollama()
    
    start_ollama_service()
    
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json()
        if not any(model["name"].startswith("llama3.2:1b") for model in models["models"]):
            pull_llama_model()
    else:
        pull_llama_model()
    
    print("Ollama LLaMA model ready!")

def get_model_choice():
    """Let user choose between HuggingFace and Ollama LLaMA."""
    print("\nChoose LLaMA model source:")
    print("1. Local Ollama LLaMA (Recommended, requires Ollama installed)")
    print("2. HuggingFace LLaMA (Requires approved access)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def setup_authentication(token: str = None):
    """Setup HuggingFace authentication either via web UI or token."""
    print("\nChecking HuggingFace authentication...")
    print("This is required to access the LLaMA model for narrative generation.")
    print("Note: You must have been granted access to Meta's LLaMA model.")
    print("      If you haven't requested access yet, visit:")
    print("      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    
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
        print("   Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
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

def load_llama(use_ollama: bool = False):
    """Load LLaMA model."""
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    if use_ollama:
        return OllamaWrapper()
    else:
        return pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

def create_questions(llm, things_to_classify: List[str], run_log: dict) -> List[str]:
    """Create all questions at once."""
    print("\nFormulating questions using LLaMA...")
    questions = []
    run_log["questions"] = []
    
    def is_valid_question(q: str) -> bool:
        """Check if the question is valid (no meta-language)."""
        bad_words = ["format", "input", "answer", "help", "please", "typical", "you"]
        return not any(word in q.lower() for word in bad_words)
    
    for thing in things_to_classify:
        print("Input prompt:", thing)
        question_log = {
            "aspect": thing,
            "attempts": []
        }
        
        if isinstance(llm, OllamaWrapper):
            max_retries = 10
            attempt = 0
            while attempt < max_retries:
                response = llm([{
                    "role": "system",
                    "content": "Format a input into a short, simple question for an image model. You return a question to further classify the image. No extra text."
                }, {
                    "role": "user",
                    "content": "here is the input, format it into a question: " + thing
                }], max_new_tokens=32, do_sample=True, temperature=0.3)
                
                question = response[0]["generated_text"][-1]["content"].strip()
                attempt_log = {
                    "attempt": attempt + 1,
                    "response": question,
                    "valid": is_valid_question(question)
                }
                question_log["attempts"].append(attempt_log)
                
                if is_valid_question(question):
                    break
                print(f"Retrying due to invalid response: {question}")
                attempt += 1
        else:
            # Full prompt for HuggingFace model
            response = llm([{
                "role": "system", 
                "content": "Format a input into a short, simple question for an image model. You return a question to further classify the image. No extra text."
            }, {
                "role": "user",
                "content": "here is the input, format it into a question (starting with 'What' and ending with 'in this image'): " + thing
            }], max_new_tokens=32, do_sample=True, temperature=0.3, pad_token_id=2)
            question = response[0]["generated_text"][-1]["content"].strip()
            question_log["attempts"].append({
                "attempt": 1,
                "response": question,
                "valid": True
            })
        
        if isinstance(llm, OllamaWrapper):
            # Clean up Ollama response if needed
            question = question.replace("Question:", "").strip()
            if not question.endswith("?"):
                question += "?"
        question += " Answer concisely, in a few words."
        questions.append(question)
        question_log["final_question"] = question
        run_log["questions"].append(question_log)
        print(f"Generated question for '{thing}': {question}")
    return questions

def classify_batch(file_path: str, things_to_classify: List[str], use_ollama: bool = False):
    """Classify multiple aspects of an image."""
    if file_path.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        raise ValueError("File type not supported. Only images are supported.")
    
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "file": os.path.basename(file_path),
        "model": "ollama" if use_ollama else "huggingface",
        "questions": [],
        "results": []
    }
    
    try:
        print("\nLoading LLaMA to generate questions...")
        llm = load_llama(use_ollama)
        questions = create_questions(llm, things_to_classify, run_log)
        
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
            answer = moondream.query(pil_image, question)["answer"].strip()
            result = {
                "aspect": thing,
                "question": question,
                "answer": answer
            }
            results.append((thing, answer))
            run_log["results"].append(result)
            print(f"Answer: {answer}")
            
        # Save run log
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/{os.path.splitext(os.path.basename(file_path))[0]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(log_filename, 'w') as f:
            json.dump(run_log, f, indent=2)
        print(f"\nRun log saved to: {log_filename}")
            
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

    # Get model choice
    model_choice = get_model_choice()
    use_ollama = (model_choice == '1')
    
    # Setup authentication/Ollama
    if use_ollama:
        setup_ollama()
    else:
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
    
    # things_to_classify = ["grass color", "time of day", "number of people, if any", 
    #                     "weather conditions", "main activity"]
    things_to_classify = [
        "grass color",
        "time of day",
        "number of people, if any",
        "weather conditions",
        "main activity",
        "path condition",
        "tree density",
        "tree arrangement",
        "tree foliage state",
        "shadow intensity",
        "terrain type",
        "visible structures",
        "sky visibility",
        "season indicators",
        "lighting quality",
        "ground cover types",
        "path curvature",
        "landscape maintenance level",
        "natural features present"
    ]
    
    for file_path in input_files:
        print("\n" + "="*70)
        print(f"Processing: {os.path.basename(file_path)}")
        print("="*70)
        try:
            results = classify_batch(file_path, things_to_classify, use_ollama)
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