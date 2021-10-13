import React, { useEffect, useState } from "react";
import GifDetails from "./GifDetails";

interface IGifResponse {
    title: string,
    url: string,
    id: string,
    embed_url: string,
    images: any
}

function GifSearch() {
    const apiKey = "SyBnCaulCHSRBuCc4dAc2VE4lb1HESJf";
    const [gifs, setGifs] = useState<IGifResponse[]>([]);
    const [inputText, setInputText] = useState("");
    const [searchText, setSearchText] = useState("welcome");  
    
    const requestUri = `https://api.giphy.com/v1/gifs/search?api_key=${apiKey}&q=${searchText}&limit=25`;
    
    const fetchGifs = async () => {
        const response = await fetch(requestUri);
        const data = await response.json();
        setGifs(data.data as IGifResponse[]);
        console.log('gifsFetched - '+ data.data);
    } // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        fetchGifs()
        // eslint-disable-next-line
    }, [searchText]);

    const onSubmit = (e: any) => {
        e.preventDefault();
        setSearchText(inputText);
        console.log(inputText);
    }


    return (<div className="app">
    <form onSubmit={onSubmit} className="search-form">
      <input className="search-bar" type="text" value={inputText} onChange={e => setInputText(e.target.value)} ></input>
      <button className="search-button" type="submit">Search</button>
    </form>
    <div className="gifs">
        {gifs.map(each => (              
          <GifDetails key={each.id} title={each.title} imageUri={each.images.original.url}></GifDetails>
        ))}            
    </div>
</div>)

}

export default GifSearch;