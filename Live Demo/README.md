# Speadle Live Demo

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

A live demo of this project is available [Here](https://speadle.web.app).

## Requirements

* Node.js: [Link](https://nodejs.org/en/download/)
* yarn: `npm install yarn`

## Installation

Clone/Download and run the following commands:

### `yarn install`

Using yarn to install

### `npm install`

Using npm to install

### API End point

Change the url of the POST request inside of `src/Pages/Home.js`

## Available Scripts

In the project directory, you can run:

### `yarn start`
### `npm run start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `yarn build`
### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

# Speadle API

Works using a Flask REST API to receive POST requests

## Changes to make
Since the host for the demo website uses https, the model needed SSL certificates. If you are just running the website locally, make the following chanes:\
* `app.run(host= '192.168.254.133', port=9000, debug=True, ssl_context=('cert.pem', 'key.pem'))`
* change to `app.run()`

## Requirements

* Developed on Python 3.9: [Link](https://www.python.org/downloads/)

## Installation

`pip install -r requirements.txt`

## Running

### Windows: `py app.py`

### Unix: `gunicorn app:app`


# Model Training
* Reference `training/train_duck.ipynb` and `training/train_wadaba.ipynb`
* Add new data to `training/data/`
* Incorporate new data, and run training
* Export model to `modelname.pkl`

# Adding Model Website/Flask

* Add model name and ID to `src/Pages/Home.js`
* Add conditional statement to `app.py` for new model id
* Load Model in `app.py`, provide new predict function if something else was used