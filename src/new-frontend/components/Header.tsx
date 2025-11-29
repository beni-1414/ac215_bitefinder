import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-forest-900 text-earth-50 p-4 shadow-lg border-b-4 border-earth-400 sticky top-0 z-50">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-earth-100 p-2 rounded-full border-2 border-earth-400">
            {/* Simple SVG Logo: Bug under magnifying glass */}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8 text-forest-800">
              <path d="M10 10m-7 0a7 7 0 1 0 14 0a7 7 0 1 0 -14 0" />
              <path d="M21 21l-6 -6" />
              <path d="M10 10m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" />
              <path d="M16 16l-1.5 -1.5" />
              <path d="M10 7v-3" />
              <path d="M6 8l-2 -2" />
              <path d="M14 8l2 -2" />
              <path d="M10 13v3" />
              <path d="M6 12l-2 2" />
              <path d="M14 12l2 2" />
            </svg>
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
