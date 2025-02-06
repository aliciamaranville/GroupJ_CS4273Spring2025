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
SMART is a tool used to analyze the contents and metadata about tweets and display this information on a map. This map aids first responders in locating emergencies and staying updated on current events. There is a ML model that is used to classify the tweets as misinformation or not. This model gets updated as the user gives feedback through either labeling the tweet as misinformation or removing the label. This allows for the model to be tailored to what the user sees as misinformation.  

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
11. Now run ```cd src``` to navigate to the src directory then run ```python api.py``` to run the server, the server will take a while to startup
12. -- currently getting error because no pickle file, emailing advisor about that
13. add pickle folder


## Table of Contents
- Description
- Technologies and Tools
- Requirements
- Setup Instructions


## Requirements 
- Having python downloaded and installed is neccessary to run the project see download instructions [here](https://www.python.org/about/gettingstarted/)

