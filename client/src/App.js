import { useState, useEffect } from 'react';
import './App.css';
import Rate from './components/Rate';

// const URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [createNewLabel, setCreateNewLabel] = useState(false);

  return (
    <div className="App">

      {createNewLabel ? 
        <Rate />
      : (
        <div className="home">
          <h1>Welcome to the Image Rating Dataset</h1>
          <button onClick={() => setCreateNewLabel(true)}>Start Rating</button>
        </div>
      )}
    </div>
  );
}

export default App;
