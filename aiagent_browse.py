import tkinter as tk
from tkinter import ttk
import schedule
import time
import threading
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import json
import logging
import sys
from mlx_lm import load, generate
from urllib.parse import urlparse, parse_qs

print("Python executable:", sys.executable)
print("Python path:", sys.path)

# Configure logging
logging.basicConfig(
    filename="ai_agent.log",  # Log file name
    level=logging.DEBUG,      # Log level (DEBUG captures all events)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Lock for thread-safe access to the model
model_lock = threading.Lock()

# Initialize the AI model
def initialize_model():
    try:
        logging.info("Initializing AI model...")
        # Load the model and tokenizer
        model, tokenizer = load("mlx-community/QwQ-32B-4bit")
        logging.info("AI model and tokenizer initialized successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error initializing AI model or tokenizer: {e}")
        raise

# Function to clear model memory
def clear_model_memory():
    try:
        logging.info("Clearing model memory...")
        global model, tokenizer
        with model_lock:  # Ensure thread-safe access
            model, tokenizer = initialize_model()  # Reinitialize the model and tokenizer
        logging.info("Model memory cleared successfully.")
    except Exception as e:
        logging.error(f"Error clearing model memory: {e}")

# Example task: Web scraping function to fetch movie data
def fetch_movies():
    try:
        logging.info("Fetching movies from Cineman...")
        url = "https://www.cineman.ch/en/showtimes/Zurich/"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        movies = []
        for movie in soup.select('.movie'):
            title = movie.select_one('.title').text.strip()
            cinema = movie.select_one('.cinema').text.strip()
            link = movie.select_one('a')['href']
            movies.append({'title': title, 'cinema': cinema, 'link': link})
        logging.info(f"Fetched {len(movies)} movies successfully.")
        return movies
    except Exception as e:
        logging.error(f"Error fetching movies: {e}")
        return []

# Function to ensure a URL has a valid scheme
def ensure_valid_url(url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            # Add 'https://' if the scheme is missing
            valid_url = f"https://{url.lstrip('/')}"
            logging.info(f"Added scheme to URL: {valid_url}")
            return valid_url
        return url
    except Exception as e:
        logging.error(f"Error ensuring valid URL for {url}: {e}")
        return url

# Function to extract the actual URL from DuckDuckGo redirect links
def extract_actual_url(redirect_url):
    try:
        parsed_url = urlparse(redirect_url)
        if "duckduckgo.com" in parsed_url.netloc and "uddg" in parse_qs(parsed_url.query):
            actual_url = parse_qs(parsed_url.query)["uddg"][0]
            actual_url = ensure_valid_url(actual_url)  # Ensure the extracted URL is valid
            logging.info(f"Extracted actual URL: {actual_url}")
            return actual_url
        return ensure_valid_url(redirect_url)  # Ensure the original URL is valid
    except Exception as e:
        logging.error(f"Error extracting actual URL from {redirect_url}: {e}")
        return ensure_valid_url(redirect_url)

# Function to perform a web search using DuckDuckGo
def search_duckduckgo(query):
    try:
        logging.info(f"Performing DuckDuckGo search for query: {query}")
        url = f"https://duckduckgo.com/html/?q={query}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.select('.result__a'):
            title = result.text.strip()
            link = result['href']
            results.append({'name': title, 'url': link})
        logging.info(f"Found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logging.error(f"Error performing DuckDuckGo search: {e}")
        return []

# Function to scrape content from a URL
def scrape_content_from_url(url):
    try:
        url = extract_actual_url(url)  # Extract and validate the actual URL
        logging.info(f"Scraping content from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        logging.info(f"Scraped {len(content)} characters from {url}")
        return content
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return ""

# Function to retrieve and scrape content from relevant sources
def retrieve_and_scrape(query, num_sources=3):
    try:
        logging.info(f"Retrieving and scraping content for query: {query}")
        search_results = search_duckduckgo(query)
        top_results = search_results[:num_sources]
        scraped_content = []
        for result in top_results:
            url = result.get('url')
            if url:
                content = scrape_content_from_url(url)
                if content:
                    scraped_content.append(content)
        logging.info(f"Retrieved and scraped content from {len(scraped_content)} sources.")
        return scraped_content
    except Exception as e:
        logging.error(f"Error retrieving and scraping content: {e}")
        return []

# Generic task execution function
def execute_task(task_name, output_format, custom_query=None, is_test=False, dont_think=False, max_tokens=1024):
    try:
        logging.info(f"Executing task: {task_name} with output format: {output_format}")
        if task_name == "Fetch Movies":
            data = fetch_movies()
        elif task_name == "Search Internet":
            query = custom_query or "movies currently showing in Zurich Switzerland"
            data = retrieve_and_scrape(query)
        else:
            data = []

        # Generate a comprehensive answer using the AI model
        if data:
            logging.info("Generating a comprehensive answer using the AI model...")
            instruction = "Don't think, just answer." if dont_think else ""
            input_text = (
                f"{instruction}\n\n"
                f"Answer the following query based on the provided context:\n\n"
                f"Query: {custom_query or task_name}\n\n"
                f"Context:\n{json.dumps(data, indent=2)}"
            )
            try:
                with model_lock:  # Ensure thread-safe access to the model
                    response = generate(model, tokenizer, input_text, max_tokens=max_tokens)
                if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
                    comprehensive_answer = response[0]["generated_text"]
                elif isinstance(response, str):
                    comprehensive_answer = response
                else:
                    raise ValueError("Unexpected response format from AI model.")
                logging.info("Comprehensive answer generated successfully.")
            except Exception as e:
                logging.error(f"Error generating comprehensive answer: {e}")
                comprehensive_answer = "Error generating a comprehensive answer. Check logs for details."
        else:
            comprehensive_answer = "No data available to generate a comprehensive answer."

        # Output the results
        if output_format == "Console":
            print(comprehensive_answer)
        elif output_format == "File":
            output_file = "output.txt"
            with open(output_file, "w") as file:
                file.write(comprehensive_answer)
            if is_test:
                logging.info(f"Test task output saved to: {output_file}")
                print(f"Test task output saved to: {output_file}")
        logging.info(f"Task {task_name} executed successfully.")
    except Exception as e:
        logging.error(f"Error executing task {task_name}: {e}")
        if is_test:
            print(f"Error executing task {task_name}. Check logs for details.")
    finally:
        clear_model_memory()  # Ensure model memory is cleared after task execution

# Function to execute tasks in a separate thread
def execute_task_in_thread(task_name, output_format, custom_query=None, is_test=False, dont_think=False, max_tokens=1024):
    def task_wrapper():
        try:
            execute_task(task_name, output_format, custom_query, is_test, dont_think, max_tokens)
            if is_test:
                logging.info("Test task executed. Check console or output file.")
                print("Test task executed. Check console or output file.")
        except Exception as e:
            logging.error(f"Error executing task in thread: {e}")
    threading.Thread(target=task_wrapper, daemon=True).start()

# Schedule tasks
def schedule_task(task_name, output_format, interval, custom_query=None, dont_think=False):
    schedule.every(interval).weeks.do(execute_task, task_name=task_name, output_format=output_format, custom_query=custom_query, dont_think=dont_think)

# GUI for task management
def create_gui():
    try:
        logging.info("Starting GUI...")
        def add_task():
            try:
                task_name = task_name_var.get()
                output_format = output_format_var.get()
                interval = int(interval_entry.get())
                custom_query = query_entry.get() if task_name == "Search Internet" else None
                dont_think = dont_think_var.get()
                max_tokens = int(max_tokens_entry.get())
                schedule_task(task_name, output_format, interval, custom_query, dont_think)
                status_label.config(text="Task scheduled successfully!")
                logging.info(f"Task added: {task_name}, Interval: {interval} weeks, Output: {output_format}, Max Tokens: {max_tokens}")
            except Exception as e:
                logging.error(f"Error adding task: {e}")
                status_label.config(text="Error scheduling task. Check logs.")

        def test_task():
            try:
                task_name = task_name_var.get()
                output_format = output_format_var.get()
                custom_query = query_entry.get() if task_name == "Search Internet" else None
                dont_think = dont_think_var.get()
                max_tokens = int(max_tokens_entry.get())

                def task_wrapper():
                    try:
                        execute_task(task_name, output_format, custom_query, is_test=True, dont_think=dont_think, max_tokens=max_tokens)
                        status_label.config(text="Test task executed. Check console or output file.")
                    except Exception as e:
                        logging.error(f"Error testing task: {e}")
                        status_label.config(text="Error testing task. Check logs.")

                threading.Thread(target=task_wrapper, daemon=True).start()
            except Exception as e:
                logging.error(f"Error testing task: {e}")
                status_label.config(text="Error testing task. Check logs.")

        root = tk.Tk()
        root.title("AI Task Scheduler")

        ttk.Label(root, text="Task Scheduler").grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(root, text="Task:").grid(row=1, column=0, padx=5, pady=5)
        task_name_var = tk.StringVar(value="Search Internet")
        task_name_dropdown = ttk.Combobox(root, textvariable=task_name_var, values=["Fetch Movies", "Search Internet"])
        task_name_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(root, text="Output:").grid(row=2, column=0, padx=5, pady=5)
        output_format_var = tk.StringVar(value="File")
        output_format_dropdown = ttk.Combobox(root, textvariable=output_format_var, values=["Console", "File"])
        output_format_dropdown.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(root, text="Interval (weeks):").grid(row=3, column=0, padx=5, pady=5)
        interval_entry = ttk.Entry(root)
        interval_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(root, text="Custom Query (optional):").grid(row=4, column=0, padx=5, pady=5)
        query_entry = ttk.Entry(root)
        query_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(root, text="Don't Think:").grid(row=5, column=0, padx=5, pady=5)
        dont_think_var = tk.BooleanVar(value=False)
        dont_think_checkbox = ttk.Checkbutton(root, variable=dont_think_var)
        dont_think_checkbox.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(root, text="Max Tokens:").grid(row=6, column=0, padx=5, pady=5)
        max_tokens_entry = ttk.Entry(root)
        max_tokens_entry.insert(0, "1024")  # Default value
        max_tokens_entry.grid(row=6, column=1, padx=5, pady=5)

        add_task_button = ttk.Button(root, text="Add Task", command=add_task)
        add_task_button.grid(row=7, column=0, columnspan=2, pady=10)

        test_task_button = ttk.Button(root, text="Test Task", command=test_task)
        test_task_button.grid(row=8, column=0, columnspan=2, pady=10)

        status_label = ttk.Label(root, text="")
        status_label.grid(row=9, column=0, columnspan=2, pady=10)

        root.mainloop()
    except Exception as e:
        logging.error(f"Error in GUI: {e}")

# Background thread for running scheduled tasks
def run_scheduler():
    try:
        logging.info("Starting task scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logging.error(f"Error in task scheduler: {e}")

if __name__ == "__main__":
    model, tokenizer = initialize_model()
    threading.Thread(target=run_scheduler, daemon=True).start()
    create_gui()
