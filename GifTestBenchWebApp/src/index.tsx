import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import Main from './components/main';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter as Router, Route } from 'react-router-dom';
import Nav from './Nav';
import GifSearch from './components/GifSearch';

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <Nav></Nav>      
      <Route path="/" exact component={GifSearch} />
      <Route path="/gifSearch" exact component={GifSearch} />
      <Route path="/upload" exact component={Main} />
    </Router>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
