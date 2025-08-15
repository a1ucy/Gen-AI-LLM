# Build Localized LLMs with Embedding Models and RAG on ARM64

This document is a step by step guide to building a personalized agent that has private knowledge. RAGFlow for 

## Applications:

* RAGFlow https://ragflow.io/
* Docker https://www.docker.com/
* Ollama https://ollama.com/

## Steps:

1. Download and install Docker and Ollama, and increase memory limit on Docker to at least 20GB.<img src="/Users/lucydesu/Desktop/Screenshot 2025-03-29 at 2.08.37 AM.png" alt="Screenshot 2025-03-29 at 2.08.37 AM" style="zoom:50%;" />

2. Selete deepseek-r1 model from Ollama site, and copy command. Feel free to try different model.<img src="/Users/lucydesu/Desktop/Screenshot 2025-03-29 at 12.45.32 AM.png" alt="Screenshot 2025-03-29 at 12.45.32 AM" style="zoom:50%;" />

3. Paste command on your terminal. After finish installing, you may test with some prompt.![Screenshot 2025-03-29 at 12.57.53 AM](/Users/lucydesu/Desktop/Screenshot 2025-03-29 at 12.57.53 AM.png)

4. Create a folder where you want RAGFlow to be and copy the command lines to clone RAGFlow.

   * ```bash
     git clone https://github.com/infiniflow/ragflow.git
     ```

5. Open **ragflow/pyproject.toml** to change xgboost version to 1.6.0.

6. Open hidden document **ragflow/docker/.env** and make few changes. 

   * Comment "RAGFLOW_IMAGE=infiniflow/ragflow:v0.17.2-slim" and uncomment "RAGFLOW_IMAGE=infiniflow/ragflow:v0.17.2". Since embedding model is required for upload knowdledge base.
   * Uncomment "MACOS=1". Since it's for MacOS system.

7. Go back to your terminal and run the command line by line. 

   * ```bash
     cd ragflow/
     uv run download_deps.py
     docker build -f Dockerfile.deps -t infiniflow/ragflow_deps .
     docker build -f Dockerfile -t infiniflow/ragflow:nightly .
     cd docker
     $ docker compose -f docker-compose-macos.yml up -d
     ```

8. Open web browser and go to **127.0.0.1** or **localhost:80**. You'll be on RAGFlow page.

9. Add Ollama from Model providers page.

<img src="/Users/lucydesu/Library/Application Support/typora-user-images/Screenshot 2025-03-29 at 1.49.48 AM.png" alt="Screenshot 2025-03-29 at 1.49.48 AM" style="zoom: 50%;" />

10. Model type = chat, model name = <copy your model's name>, base url = http://host.docker.internal:11434, Max Tokens = 99999.

    <img src="/Users/lucydesu/Library/Application Support/typora-user-images/Screenshot 2025-03-29 at 1.54.57 AM.png" alt="Screenshot 2025-03-29 at 1.54.57 AM" style="zoom: 67%;" />

11. Open System Model Settings on the top right corner to set up embedding model.

<img src="/Users/lucydesu/Library/Application Support/typora-user-images/Screenshot 2025-03-29 at 2.04.15 AM.png" alt="Screenshot 2025-03-29 at 2.04.15 AM" style="zoom:67%;" />

12. Go to Knowledge Base on top left and click on Create knowledge base.
13. After created the knowledge base name, add files into the dataset and parsing the data by click on run button next to **PENDING** text.<img src="/Users/lucydesu/Desktop/Screenshot 2025-03-29 at 2.12.35 AM.png" alt="Screenshot 2025-03-29 at 2.12.35 AM" style="zoom:67%;" />
14. Once parsing completed, go to Chat to and create an agent to try to new information.<img src="/Users/lucydesu/Library/Application Support/typora-user-images/Screenshot 2025-03-29 at 2.16.44 AM.png" alt="Screenshot 2025-03-29 at 2.16.44 AM" style="zoom:50%;" />

## Result:

![Screenshot 2025-03-29 at 2.22.19 AM](/Users/lucydesu/Library/Application Support/typora-user-images/Screenshot 2025-03-29 at 2.22.19 AM.png)

![Screenshot 2025-03-29 at 2.21.25 AM](/Users/lucydesu/Desktop/Screenshot 2025-03-29 at 2.21.41 AM.png)

## Q&A:

* Q: Parsing Status shows Fail / stuck on searching after prompted.

  A: Increase Docker memory limit.

* Q: Model not found when adding Ollama.

  A: Model name has to be exactly same from command copied from Ollama site.

* Q: Embedding model is empty.

  A: Download desired embedding model from Ollama site. (Similar to step 2)



