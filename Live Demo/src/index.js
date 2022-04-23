import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();


// Import the functions you need from the SDKs you need

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDnlWQFdAksgtpMURvv8L3rgYYebOsEuLc",
  authDomain: "speadle.firebaseapp.com",
  projectId: "speadle",
  storageBucket: "speadle.appspot.com",
  messagingSenderId: "923353639749",
  appId: "1:923353639749:web:b7c13c34b53fc0e9e2ac2c",
  measurementId: "G-HV4K5DYJ04"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
getAnalytics(app);