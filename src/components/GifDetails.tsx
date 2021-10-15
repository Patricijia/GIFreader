import React from 'react';
import './GifDetails.css';
import {Link} from 'react-router-dom';

interface IGifDetailProps {
    title: string,
    imageUri: string    
}

function GifDetails({title, imageUri}: IGifDetailProps) {
    return (
        <div className="gif" >
                {/* <h3>{title}</h3> */}
                <img tabIndex={0} src={imageUri} alt ='' ></img>            
        </div>
    );
}

export default GifDetails;