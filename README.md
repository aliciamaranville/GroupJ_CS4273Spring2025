# GroupJ_CS4273Spring2025

## Members
Mahmoud Mousa Hamad - Mentor  
Vishnu Patel - Product Owner  
Aidan Foreman - Sprint Master 1  
Mark Castle - Sprint Master 2  
Alicia Maranville - Sprint Master 3  
Jacob Wells - Sprint Master 4  
Thorpe Mayes - Quality Assurance

## Description
SMART is a tool used to analyze the contents and metadata about tweets and display this information on a map to be able to track information in real time. This map has aided first responders in emergencies by being able to monitor real time data and has even been used during the presidential inauguration. The repo is a microservice that contains a ML model that is trained to classify tweets as misinformation or not. This model gets updated as the user gives feedback through either labeling the tweet as misinformation or removing the label. This allows for the model to be tailored to what the user sees as misinformation.  
Click [here](https://www.ou.edu/disc/initiatives/tools/smart) for a high level overview

## Technologies and Tools

- Python
- FastAPI
- pip and virtual environments for dependency management
- Hashmaps
- Machine learning models
- Web interface for map display

## Goals and Progress Plan 
Weekly progress update meetings with mentor - Thursdays 2-2:30p via Microsoft Teams

- [] Understand the existing code base  
  Expected 1/30/25
- [] Run and demo the existing code base  
  Expected 1/30/25
- [] Document the existing code base  
  Expected 3/13/25
- [] Add way for there to be multiple users all with their own models  
  Expected 4/10/25
- [] Test that the program works with multiple users  
  Expected 5/1/25

## Setup Instruction

1. Make sure you have access to the OU DISC SMART misinformation repo
2. Navigate to the repo page and click the green code button and copy the url
3. Navigate to your IDE terminal and run ```git clone <url>``` where <url> is the copied url from step 2
5. Now in order to run the code a virtual environment will be needed and the libraries downloaded
6. Navigate to the cloned repo and run ```python3 -m venv <name_of_virtual_environment>``` this will create the virtual environment
7. Run ```source <name_of_virtual_environment>/bin/activate``` to activate the virtual environment.
8. In VsCode you can see (name_of_virtual_environment) in the terminal which lets you know its activated.
9. Then run ```pip install -r requirements.txt``` or ```python -m pip install -r requirements.txt``` which will download all the necessaary libraries.
10. Lastly before running we will need to copy the environment variables, make a file named .env, then copy everything from the .env.example file into the .env file
11. Now we will also need to create a folder named 'pickles' inside the SMARTMisinformation folder.
12. Now run ```cd src``` to navigate to the src directory then run ```python api.py``` to run the server, the server will take a while to startup.
13. Navigate to localhost:9090, and check that it says "Welcome to the tweet classification model"
14. You're all set up!


## Table of Contents
- Description
- Technologies and Tools
- Requirements
- Setup Instructions


## Requirements 
- Having python 3.7 - 3.12 downloaded and installed is neccessary to run the project see download instructions [here](https://www.python.org/about/gettingstarted/)
- The project may not run with python 3.13+ since DistUtils is a necessary package but is not part of python 3.13 onwards.
- As well a working understanding of python and web frameworks for building API's will be necessary to understand the code for more information about FastAPI click [here](https://fastapi.tiangolo.com/)
- An understanding of virtual environments in python see more information [here](https://docs.python.org/3/library/venv.html)


