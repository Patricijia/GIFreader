import React from 'react';
import './GifDetails.css';
import {Link} from 'react-router-dom';

interface IGifDetailProps {
    title: string,
    imageUri: string    
}

function GifDetails({title, imageUri}: IGifDetailProps) {
    return (
        <div className="gif">
            <Link to={`#`}>
                {/* <h3>{title}</h3> */}
                <img src={imageUri} alt ='' ></img>
            </Link>
        </div>
    );
}

export default GifDetails;