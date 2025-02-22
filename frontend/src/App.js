import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { useDropzone } from 'react-dropzone';

function App() {
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([
    {
      role: 'assistant',
      content: "Hello! I'm your AI Academic Advisor, and I'm excited to help you with any questions about the CSE curriculum here at OSU! Feel free to ask away!",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const [uploadedFiles, setUploadedFiles] = useState([]); // State for uploaded files
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    const newMessage = { role: 'user', content: message };
    setConversation([...conversation, newMessage]);

    setMessage('');
    setLoading(true);

    try {
      const res = await axios.post('http://127.0.0.1:5000/chat', { message });
      const aiMessage = { role: 'assistant', content: res.data.response };
      setConversation((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error fetching the AI response', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      setUploadedFiles(acceptedFiles);
    },
  });

  const handleUpload = async () => {
    if (uploadedFiles.length === 0) return; // Prevent upload if no files

    const formData = new FormData();
    formData.append('file', uploadedFiles[0]);
    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      const aiMessage = { role: 'assistant', content: response.data.response };
      setConversation((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error uploading the file:', error);
    }
    setUploadedFiles([]); // Clear uploaded files after upload
    setMessage('');
  };

  const uploadedFileName = uploadedFiles.length > 0 ? uploadedFiles[0].name : 'Upload your transcript here';

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  const ThemeIcon = () => (
    theme === 'light' ? (
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M12 3V4M12 20V21M4 12H3M6.31412 6.31412L5.5 5.5M17.6859 6.31412L18.5 5.5M6.31412 17.69L5.5 18.5M17.6859 17.69L18.5 18.5M21 12H20M16 12C16 14.2091 14.2091 16 12 16C9.79086 16 8 14.2091 8 12C8 9.79086 9.79086 8 12 8C14.2091 8 16 9.79086 16 12Z"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ) : (
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M3.32031 11.6835C3.32031 16.6541 7.34975 20.6835 12.3203 20.6835C16.1075 20.6835 19.3483 18.3443 20.6768 15.032C19.6402 15.4486 18.5059 15.6835 17.3203 15.6835C12.3497 15.6835 8.32031 11.6541 8.32031 6.68353C8.32031 5.49797 8.55517 4.36367 8.97181 3.32706C5.65957 4.65561 3.32031 7.89639 3.32031 11.6835Z"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    )
  );

  return (
    <div className="chat-container">
      <button className="theme-toggle" onClick={toggleTheme}>
        <ThemeIcon />
        <span>{theme === 'light' ? 'Dark' : 'Light'}</span>
      </button>
      <h1>AdvisorAI</h1>
      <div className="chat-box">
        <div className="chat-history">
          {conversation.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              <p>{msg.content}</p>
            </div>
          ))}
          {loading && (
            <div className="chat-message assistant loading">
              <span className="ellipsis"></span>
              <span className="ellipsis"></span>
              <span className="ellipsis"></span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Render only the Upload File button styled like the Upload Transcript button */}
        <div className={`drag-drop-container ${isDragActive ? 'drag-active' : ''}`} {...getRootProps()}>
          <input {...getInputProps()} />
          <div onClick={handleUpload}>
            <p>{uploadedFileName}</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="chat-input">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            className="message-input"
            placeholder="Ask something..."
          />
          <button type="submit" className="send-btn">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;