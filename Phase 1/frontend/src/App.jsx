import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import RoleSelect from './pages/RoleSelect';
import Interview from './pages/Interview';
import Feedback from './pages/Feedback';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/select" element={<RoleSelect />} />
      <Route path="/interview" element={<Interview />} />
      <Route path="/feedback" element={<Feedback />} />
    </Routes>
  );
}

export default App;
