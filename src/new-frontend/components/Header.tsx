import React from 'react';
import bitefinderLogo from './bitefinder-logo.png';

const Header: React.FC = () => {
  return (
    <header className="bg-forest-900 text-earth-50 p-4 shadow-lg border-b-4 border-earth-400 sticky top-0 z-50">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-earth-100 p-2 rounded-full border-2 border-earth-400 overflow-hidden">
            <img
              src={bitefinderLogo}
              alt="BiteFinder logo"
              className="w-11 h-11 object-cover scale-110"
            />
          </div>
          <div>
            <h1 className="text-2xl font-serif font-bold tracking-wide text-earth-100">BiteFinder</h1>
            <p className="text-xs text-forest-300 uppercase tracking-widest font-semibold">Wilderness Guide</p>
          </div>
        </div>
        <div className="hidden sm:block">
          <span className="text-sm font-medium bg-forest-800 px-3 py-1 rounded-full text-forest-200 border border-forest-600">
            v1.0 • Ranger Rick AI
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
