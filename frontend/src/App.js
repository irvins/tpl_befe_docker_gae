import React, { useEffect, useState } from 'react';
import axios from 'axios';

import logo from './logo.svg';
import './App.css';


function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const endpoint = process.env.REACT_APP_BACKEND_URL + "/openai_test"
        const response = await axios.get(endpoint);
        setData(response.data);
      } catch (error) {
        console.error(error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="App">
      {/* Your component JSX */}
      {/* Render the data received from the backend */}
      {data && <p>Data received {process.env.REACT_APP_BACKEND_URL}: {data}</p>}
    </div>
  );
}

export default App;
