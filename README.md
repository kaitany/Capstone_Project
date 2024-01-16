
# Customer-Churn-Prediction-Model
This challenge is for an African telecommunications company that provides customers with airtime and mobile data bundles. The objective of this challenge is to develop a machine learning model to predict the likelihood of each customer “churning,” i.e. becoming inactive and not making any transactions for 90 days.
# Model Deployment with FastAPI and Docker

[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![MIT licensed](https://img.shields.io/badge/license-mit-blue?style=for-the-badge&logo=appveyor)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

## Introduction


This repository demonstrates the deployment of a machine learning model using FastAPI within a Docker container. The project provides a RESTful API endpoint to interact with the model predictions.


## Prerequisites

Ensure you have the following installed:

- [Docker](https://www.docker.com/)
- Python 3.x



## Setup

Install the required packages to be able to run the evaluation locally.

You need to have [Python 3](https://www.python.org/) on your system (*a Python version greater than 3.8*). Then you can clone this repo and being at the repo's root :: repository_name> ...  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

The both long command-lines have a same structure, they pipe multiple commands using the symbol ` ; ` but you may manually execute them one after another.

1. *Create the Python's virtual environment* that isolates the required libraries of the project to avoid conflicts;
2. *Activate the Python's virtual environment* so that the Python kernel & libraries will be those of the isolated environment;
3. *Upgrade Pip, the installed libraries/packages manager* to have the up-to-date version that will work correctly;
4. *Install the required libraries/packages* listed in the requirements.txt file so that it will be allow to import them into the python's scripts and notebooks without any issue.

*NB:* For MacOs users, please install Xcode if you have an issue.

## Usage

1. Start the FastAPI server:

- Run the demo apps (being at the repository root):
        
  FastAPI:
    

    <!-- - churn prediction

          uvicorn main:app --reload  -->


  - Go to your browser at the following address, to explore the api's documentation :
        
  http://localhost:8000/docs


- Screenshots

![FastAPI Logo](./image/fastapi.png)
![streamlit Logo](./image/streamlit.png)

2. Docker Deployment

- Build the Docker image:
docker build -t your_image_name .

- Run the Docker container:
docker run -d -p 8000:800 your_image_name

This will run the container in detached mode, exposing it on port 80.

- Access the API:
Open your web browser or use tools like curl or Postman to access the API at http://localhost.

## API Endpoints
/predict: Endpoint to make predictions using the model. Send get requests with necessary data to the fast api for  predictions to the gradio interface.

## Project Structure
- main.py: Contains the FastAPI application files.
- app.py/: contain the streamlit application.
- requirements.txt: Lists Python dependencies.
- Dockerfile.fastapi: Contains instructions to build the Docker image of the fastapi.
- Dockerfile.streamlit: Contains instructions to build the Docker image of the streamlit.
- Docker-compose.yml: Contains instructions to build the Docker image of both the  fastapi and the streamlit .


## Resources
Here are some ressources you would read to have a good understanding of FastAPI :
- [Tutorial - User Guide](https://fastapi.tiangolo.com/tutorial/)
- [Video - Building a Machine Learning API in 15 Minutes ](https://youtu.be/C82lT9cWQiA)
- [FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)
- [Video - Deploy ML models with FastAPI, Docker, and Heroku ](https://www.youtube.com/watch?v=h5wLuVDr0oc)
- [FastAPI Tutorial Series](https://www.youtube.com/watch?v=tKL6wEqbyNs&list=PLShTCj6cbon9gK9AbDSxZbas1F6b6C_Mx)
- [Http status codes](https://www.linkedin.com/feed/update/urn:li:activity:7017027658400063488?utm_source=share&utm_medium=member_desktop)


## Contributing

Feel free to make a PR or report an issue 😃.

## License

This project is licensed under the MIT License.

Oh, one more thing, please do not forget to put a description when you make your PR 🙂.

## Author

- Team Santorini[Benedicta,Benjamin,Seth,Phonix]
