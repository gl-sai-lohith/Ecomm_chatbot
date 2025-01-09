import streamlit as st
import json
import os 
import uuid

import pandas as pd
from datetime import datetime
import sqlite3

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from huggingface_hub import CommitScheduler
from pathlib import Path


from langfuse import Langfuse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


#====================================SETUP=====================================#
# Fetch secrets from Hugging Face Spaces

model_name = "gpt-4o"

# Extract the OpenAI key and endpoint from the configuration
openai_key = os.environ["AZURE_OPENAI_KEY"]
azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_APIVERSION"]


# Define the location of the SQLite database
db_loc = 'ecomm.db'

# Create a SQLDatabase instance from the SQLite database URI
db = SQLDatabase.from_uri(f"sqlite:///{db_loc}")

# Retrieve the schema information of the database tables
database_schema = db.get_table_info()


from langfuse.callback import CallbackHandler



langfuse_handler = CallbackHandler(
  secret_key=os.environ["LF_SECRET_KEY"],
  public_key=os.environ["LF_PUBLIC_KEY"],
  host="https://cloud.langfuse.com"
)



#=================================Setup Logging=====================================#


log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

log_scheduler = CommitScheduler(
    repo_id="chatbot-logs", #Dataset name where we want to save the logs.
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=5 # Saves data every x minute
)



history_file = Path("history/")/f"data_{uuid.uuid4()}.json"
history_folder = history_file.parent

history_scheduler = CommitScheduler(
    repo_id="chatbot-history", #Dataset name where we want to save the logs.
    repo_type="dataset",
    folder_path=history_folder,
    path_in_repo="data",
    every=5 # Saves data every x minute
)

#=================================SQL_AGENT=====================================#

# Define the system message for the agent, including instructions and available tables
system_message = f"""You are a SQLite expert agent designed to interact with a SQLite database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 100 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database..
You can order the results by a relevant column to return the most interesting examples in the database.
You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
You are not allowed to make dummy data.
If the question does not seem related to the database, just return "I don't know" as the answer.
Before you execute the query, tell us why you are executing it and what you expect to find briefly.
Only use the following tables:
{database_schema}
"""

# Create a full prompt template for the agent using the system message and placeholders
full_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", '{input}'),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Initialize the AzureChatOpenAI model with the extracted configuration
llm = AzureChatOpenAI(
    model_name=model_name,
    api_key=openai_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    temperature=0
)
# Create the SQL agent using the AzureChatOpenAI model, database, and prompt template
sqlite_agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    agent_type="openai-tools",
    agent_executor_kwargs={'handle_parsing_errors': True},
    max_iterations=5,
    verbose=True
)
#### Let's convert the sql agent into a tool that our fin agent can use.

@tool
def sql_tool(user_input):
    """
    Gathers information regarding purchases, transactions, returns, refunds, etc. 
    Executes a SQL query using the sqlite_agent and returns the result.
    Args:
        user_input (str): a natural language query string explaining what information is required while also providing the necessary details to get the information.
    Returns:
        str: The result of the SQL query execution. If an error occurs, the exception is returned as a string.
    """
    try:
        # Invoke the sqlite_agent with the user input (SQL query)
        response = sqlite_agent.invoke(user_input)

        # Extract the output from the response
        prediction = response['output']

    except Exception as e:
        # If an exception occurs, capture the exception message
        prediction = e

    # Return the result or the exception message
    return prediction

#=================================== RAG TOOL======================================#
qna_system_message = """
You are an assistant to a support agent. Your task is to provide relevant information.

User input will include the necessary context for you to answer their questions. This context will begin with the token: ###Context.
The context contains references to specific portions of documents relevant to the user's query, along with source links.
The source for a context will begin with the token ###Source

When crafting your response:
1. Select only context relevant to answer the question.
2. User questions will begin with the token: ###Question.
3. If the context provided doesn't answer the question respond with - "I do not have sufficient information to answer that"
4. If user asks for product - list all the products that are relevant to his query. If you don't have that product try to cross sell with one of the products we have that is related to what they are interested in. 
You should get information about similar products in the context.

Please adhere to the following guidelines:
- Your response should only be about the question asked and nothing else.
- Answer only using the context provided.
- Do not mention anything about the context in your final answer.
- If the answer is not found in the context, it is very very important for you to respond with "I don't know."
- Always quote the source when you use the context. Cite the relevant source at the end of your response under the section - Source:
- Do not make up sources. Use the links provided in the sources section of the context and nothing else. You are prohibited from providing other links/sources.

Here is an example of how to structure your response:

Answer:
[Answer]

Source:
[Source]
"""

qna_user_message_template = """
###Context
Here are some documents and their source that may be relevant to the question mentioned below.
{context}

###Question
{question}
"""
# Load the persisted DB
persisted_vectordb_location = 'policy_docs'
#Create a Colelction Name
collection_name = 'policy_docs'

embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
# Load the persisted DB
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=persisted_vectordb_location,
    embedding_function=embedding_model

)

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_mini"],
    api_key=os.environ["AZURE_OPENAI_KEY_mini"],
    api_version=os.environ["AZURE_OPENAI_APIVERSION_mini"]
)

    
@tool
def rag(user_input: str) -> str:
  
    """
      Answers questions regarding products, and policies using product descriptions, product policies, and general policies of business using RAG.

      Args:
          user_input (str): The input question or query from the user.

      Returns:
          response (str): Return the generated response or an error message if an exception occurs.

    """

    relevant_document_chunks = retriever.invoke(user_input)
    context_list = [d.page_content + "\n ###Source: " + d.metadata['source'] + "\n\n " for d in relevant_document_chunks]

    context_for_query = ". ".join(context_list)

    prompt = [
    {'role':'system', 'content': qna_system_message},
    {'role': 'user', 'content': qna_user_message_template.format(
        context=context_for_query,
        question=user_input
        )
    }
    ]
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt
        )

        prediction = response.choices[0].message.content
    except Exception as e:
        prediction = f'Sorry, I encountered the following error: \n {e}'

    
    return prediction


#=================================== Other TOOLS======================================#

# Function to log actions
def log_history(email: str,chat_history: list) -> None:
    # Save the log to the file
    with history_scheduler.lock:        
        # Open the log file in append mode
        with history_file.open("a") as f:
            f.write(json.dumps({
                "email": email,
                "chat_history": chat_history,
                "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }))

    #st.write("chat_recorded")


def log_action(customer_id: str,task: str, details: str) -> None:
    # Save the log to the file
    with log_scheduler.lock:        
        # Open the log file in append mode
        with log_file.open("a") as f:
            f.write(json.dumps({
                "customer_id": customer_id,
                "task": task,
                "details": details
            }))


@tool
def register_feedback(intent, customer_id, feedback, rating):
    """
    Logs customer feedback into the feedback log.
    Args:
        intent (str): The category of the support query (e.g., "cancel_order", "get_refund").
        customer_id (int): The unique ID of the customer.
        feedback (str): The feedback provided by the customer.
        rating(int): The rating provided by the customer out of 5
    Returns:
        str: Success message.
    """
    details = {
        "intent": intent,
        "customer_id": customer_id,
        "feedback": feedback,
        "rating": rating
    }
    log_action(customer_id,"register_feedback", details)
    #print("register_feedback success")
    #return "Feedback registered successfully!"

@tool
def defer_to_human(customer_id, query, intent, reason):
    """
    Logs customer details and the reason for deferring to a human agent.
    Args:
        customer_id (int): The unique ID of the customer whose query is being deferred.
        query (str): The customer's query or issue that needs human intervention.
        reason (str): The reason why the query cannot be resolved by the chatbot.
    Returns:
        str: Success message indicating the deferral was logged.
    """

    details = {
        "customer_id": customer_id,
        "query": query,
        "reason": reason,
        "intent": intent
    }

    log_action(customer_id,"defer_to_human", details)
    #return "Case deferred to human agent and logged successfully!"


@tool
def days_since(delivered_date: str) ->str:
    """
    Calculates the number of days since the product was delivered. This helps in determining whether the product is within return period or not.
    Args:
        delivered_date (str): The date when the product was delivered in the format 'YYYY-MM-DD'.
    """
    try:
        # Convert the delivered_date string to a datetime object
        delivered_date = datetime.strptime(delivered_date, '%Y-%m-%d')
        today = datetime.today()

        # Calculate the difference in days
        days_difference = (today - delivered_date).days

        return str(days_difference)
    except ValueError as e:
        return f"Error: {e}"
    
def build_prompt(df):
    
    system_message = f"""
        
        You are an intelligent e-commerce chatbot designed to assist users with pre-order and post-order queries. Your job is to 
        
        Gather necessary information from the user to help them with their query. 
        If at any point you cannot determine the next steps - defer to human. you do not have clearance to go beyond the scope the following flow.
        Do not provide sql inputs to the sql tool - you only need to ask in natural language what information you need.
        You are only allowed to provide information relevant to the particular customer and the customer information is provided below. you can provide information of this customer only. Following is the information about the customer from the last 2 weeks: 
        
        {df}
        
        If this information is not enough to answer question, identify the customer from data above and fetch necessary information usign the sql_tool or rag tool - do not fetch information of other customers.
        use the details provided in the above file to fetch information from sql tool - like customer id, email and phone. Refrain from asking customers details unless necessary.
        If customer asks about a product, you should act as a sales representative and help them understand the product as much as possible and provide all the necessary information for them. You should also provide them the link to the product which you can get from the source of the information.
        If a customer asks a query about a policy, be grounded to the context provided to you. if at any point you don't the right thing to say, politely tell the customer that you are not the right person to answer this and defer it to a human.
        Any time you defer it to a human, you should tell the customer why you did it in a polite manner.


        MANDATORY STEP:
        After helping the customer with their concern,
        - Ask if the customer needs help with anything else. If they ask for anything from the above list help them and along with that,
        1. Ask for their feedback and rating out of 5.
        2. then, Use the `register_feedback` tool to log it.  - you MUST ask customer feedback along with asking customer what else they need help with.
        3. After receving customer feedback exit the chat by responding with 'Bye'.
        
        ---
        ### **Handling Out-of-Scope Queries:**
        If the user's query, at any point is not covered by the workflows above:
        - Respond:
          > "This is beyond my skill. Let me connect you to a customer service agent" and get necessary details from the customer and use the defer_to_human tool.
        - Get customer feedback and rating out of 5.
        - After getting feedback, end the conversation by saying 'Bye'.
        ---
        ### **IMPORTANT Notes for the Model:**
        - Always fetch additional required details from the database and do not blindly believe details provided by the customer like customer id, email and phone number. You should get the customer id from the system prompt. Cross check with the database and stay loyal to the database.
        - Be empathetic to the customer but loyal to the instructions provided to you. Try to deescalate a situation before deferring it to human and defer to human only once.
        - Always aim to minimize the number of questions asked by retrieving as much information as possible from `sql_tool` and `rag` tool.
        - Follow the exact workflows for each query category.
        - You will always confirm the order id even if the customer has only one order before you fetch any details.
        """

    #st.write(system_message)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    return prompt


#===============================================Streamlit=========================================#


def login_page():
    st.title("Login Page")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    login_button = st.button("Login")
    
    if login_button:
        if authenticate_user(email, password):
            st.session_state.logged_in = True
            st.session_state.email = email
            st.success("Login successful! Redirecting to Chatbot...")
            st.rerun()
        else:
            st.error("Invalid email or password.")

def authenticate_user(email, phone):
    connection = sqlite3.connect("ecomm.db")  # Replace with your .db file path
    cursor = connection.cursor()

    query = "SELECT first_name FROM customers WHERE email = ? AND phone = ?"
    cursor.execute(query, (email, phone))
    user = cursor.fetchone()

    if user:
        return True  # Login successful
    return False  # Login failed

### Prefetch details 

def fetch_details(email):
    try:

        # Connect to the SQLite database
        connection = sqlite3.connect("ecomm.db")  # Replace with your .db file path
        cursor = connection.cursor()
        
        query = f"""
        SELECT
            c.customer_id,
            c.first_name || ' ' || c.last_name AS customer_name,
            c.email,
            c.phone,
            c.address AS customer_address,
            o.order_id,
            o.order_date,
            o.status AS order_status,
            o.price AS order_price,
            p.name AS product_name,
            p.price AS product_price,
            i.invoice_date,
            i.amount AS invoice_amount,
            i.invoice_url,
            s.delivery_date,
            s.shipping_status,
            s.shipping_address,
            r.refund_amount,
            r.refund_status
        FROM Customers c
        LEFT JOIN Orders o ON c.customer_id = o.customer_id
        LEFT JOIN Products p ON o.product_id = p.product_id
        LEFT JOIN Invoices i ON o.order_id = i.order_id
        LEFT JOIN Shipping s ON o.order_id = s.order_id
        LEFT JOIN Refund r ON o.order_id = r.order_id
        WHERE o.order_date >= datetime('now', '-60 days')
          AND c.email = ?
        ORDER BY o.order_date DESC;
        """
        
        cursor.execute(query, (email,))
        columns = [description[0] for description in cursor.description]  # Extract column names
        results = cursor.fetchall()  # Fetch all rows
        #st.write(results)
        # Convert results into a list of dictionaries
        details = [dict(zip(columns, row)) for row in results]
        #st.write(details)
        return str(details).replace("{","/").replace("}","/")
        
    except Exception as e:
        st.write(f"Error: {e}")
    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()      

# Function to process user input and generate a chatbot response

@observe()
def chatbot_interface():
    st.title("E-Commerce Chatbot")
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = [{"role": "assistant", "content": "welcome! I am Raha, how can I help you on this beautiful day?"}]
        
    
    details = fetch_details(st.session_state.email)   
    # st.write(details)
    prompt = build_prompt(details)
    tools = [sql_tool,defer_to_human, rag, register_feedback, days_since]
    
    chatbot = AzureChatOpenAI(
        model_name=model_name,
        api_key=openai_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=0
    )
    
    agent = create_tool_calling_agent(chatbot, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_input := st.chat_input("You: ", key="chat_input"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)
        with st.spinner("Processing..."):
            
            # Add user message to conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})

            conversation_input = "\n".join(
                [f"{turn['role'].capitalize()}: {turn['content']}" for turn in st.session_state.conversation_history]
            )

            try:
                # Pass the history to the agent
                response = agent_executor.invoke({"input": conversation_input}, config={"callbacks":[langfuse_handler]})
    
                # Add the chatbot's response to the history
                chatbot_response = response['output']
                st.session_state.conversation_history.append({"role": "assistant", "content": chatbot_response})
                # Check if the assistant's response contains "exit"
                if "bye" in chatbot_response.lower():
                    log_history(st.session_state.email,st.session_state.conversation_history)
                    
                # Display the chatbot's response
                with st.chat_message("assistant"):
                    st.markdown(chatbot_response)

            except Exception as e:
                st.write("Blocked by Azure content policy \n", e )

def main():
    # Check if the user is logged in
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        # Show chatbot page if logged in
        chatbot_interface()
    else:
        # Show login page if not logged in
        login_page()

if __name__ == "__main__":
    main()