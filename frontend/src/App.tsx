import React, { useState, useEffect } from 'react';
import Chessboard from 'chessboardjsx';
import axios from 'axios';
import { DndProvider } from 'react-dnd/dist';
import { HTML5Backend } from 'react-dnd-html5-backend';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function App() {
  const [fen, setFen] = useState(INITIAL_FEN);
  const [isThinking, setIsThinking] = useState(false);
  const [evaluation, setEvaluation] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playerColor, setPlayerColor] = useState<'white' | 'black' | null>(null);
  const [gameStarted, setGameStarted] = useState(false);

  // Add initial connection test
  useEffect(() => {
    const testConnection = async () => {
      try {
        console.log('Testing connection to backend at:', API_URL);
        const response = await axios.get(API_URL);
        console.log('Backend connection successful:', response.data);
      } catch (err) {
        console.error('Failed to connect to backend:', err);
        setError('Failed to connect to chess engine. Please check if the backend is running.');
      }
    };
    testConnection();
  }, []);

  // Start engine move if computer plays white
  useEffect(() => {
    if (gameStarted && playerColor === 'black') {
      getEngineMove();
    }
  }, [gameStarted]);

  const startGame = (color: 'white' | 'black') => {
    setPlayerColor(color);
    setGameStarted(true);
    setFen(INITIAL_FEN);
    setError(null);
    setEvaluation(null);
  };

  const isPlayerTurn = () => {
    const isWhiteTurn = fen.includes(' w ');
    return (playerColor === 'white' && isWhiteTurn) || (playerColor === 'black' && !isWhiteTurn);
  };

  const onDrop = async ({ sourceSquare, targetSquare, piece }: { sourceSquare: string; targetSquare: string; piece: string }) => {
    if (!gameStarted) {
      setError('Please select your color to start the game');
      return false;
    }

    if (!isPlayerTurn()) {
      setError("It's not your turn");
      return false;
    }

    try {
      setError(null);
      const move = `${sourceSquare}${targetSquare}`;
      console.log('Making move:', move);
      const response = await axios.post(`${API_URL}/move`, {
        fen,
        move,
        time_limit: 1.0
      });
      
      if (response.data.success) {
        console.log('Move successful:', response.data);
        setFen(response.data.fen);
        // Evaluate position after player's move
        evaluatePosition();
        // Make engine move after player's move
        setTimeout(getEngineMove, 500);
        return true;
      } else {
        console.error('Invalid move:', response.data);
        setError('Invalid move');
        return false;
      }
    } catch (error) {
      console.error('Error making move:', error);
      setError('Failed to make move');
      return false;
    }
  };

  const getEngineMove = async () => {
    if (!gameStarted) {
      setError('Please select your color to start the game');
      return;
    }

    if (isPlayerTurn()) {
      setError("It's your turn to move");
      return;
    }

    setIsThinking(true);
    setError(null);
    try {
      console.log('Requesting engine move');
      const response = await axios.post(`${API_URL}/engine_move`, {
        fen,
        time_limit: 1.0
      });
      
      if (response.data.success) {
        console.log('Engine move successful:', response.data);
        setFen(response.data.fen);
        // Evaluate position after engine's move
        evaluatePosition();
      } else {
        console.error('Engine move failed:', response.data);
        setError(response.data.error || 'Engine failed to make a move');
      }
    } catch (error) {
      console.error('Error getting engine move:', error);
      setError('Failed to get engine move');
    }
    setIsThinking(false);
  };

  const evaluatePosition = async () => {
    if (!gameStarted) {
      setError('Please select your color to start the game');
      return;
    }

    setError(null);
    try {
      console.log('Evaluating position');
      const response = await axios.get(`${API_URL}/evaluate`, {
        params: { fen, time_limit: 1.0 }
      });
      console.log('Evaluation result:', response.data);
      setEvaluation(response.data.score);
    } catch (error) {
      console.error('Error evaluating position:', error);
      setError('Failed to evaluate position');
    }
  };

  // Add initial evaluation when game starts
  useEffect(() => {
    if (gameStarted) {
      evaluatePosition();
    }
  }, [gameStarted]);

  if (!gameStarted) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>Chess Engine</h1>
        </header>
        <main>
          <div className="color-selection">
            <h2>Choose your color</h2>
            <div className="color-buttons">
              <button onClick={() => startGame('white')}>Play as White</button>
              <button onClick={() => startGame('black')}>Play as Black</button>
            </div>
          </div>
          {error && <div className="error">{error}</div>}
        </main>
      </div>
    );
  }

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="App">
        <header className="App-header">
          <h1>Chess Engine</h1>
        </header>
        <main>
          <div className="chess-board">
            <Chessboard
              position={fen}
              onDrop={onDrop}
              orientation={playerColor || 'white'}
              width={400}
              draggable={isPlayerTurn()}
              boardStyle={{
                borderRadius: '4px',
                boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)'
              }}
            />
          </div>
          <div className="controls">
            <div className="status">
              Playing as: {playerColor}
              {isPlayerTurn() ? " - Your turn" : " - Engine's turn"}
            </div>
            <button onClick={getEngineMove} disabled={isThinking || isPlayerTurn()}>
              {isThinking ? 'Thinking...' : 'Engine Move'}
            </button>
            <button onClick={evaluatePosition}>
              Evaluate Position
            </button>
            <button onClick={() => startGame(playerColor || 'white')}>
              New Game
            </button>
            {evaluation !== null && (
              <div className="evaluation">
                Evaluation: {(evaluation / 100).toFixed(2)}
              </div>
            )}
            {error && (
              <div className="error">
                {error}
              </div>
            )}
          </div>
        </main>
      </div>
    </DndProvider>
  );
}

export default App; 