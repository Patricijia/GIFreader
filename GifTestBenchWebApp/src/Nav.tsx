import React from "react";
import "./index.css";
import { Link } from 'react-router-dom';

function Nav() {
    const navStyle = {
        color: 'white',
        textDecoration: 'none'
    };
  return (
    <nav>
        <h3>Gif Description Generator</h3>
        <ul className="nav-links">
            <Link style={navStyle} to="/gifSearch">
                <li>Home</li>
            </Link>
            <Link style={navStyle} to="/upload">
                <li>Upload Custom Image</li>
            </Link>           
        </ul>
    </nav>
  );
}

export default Nav;
