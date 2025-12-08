import React, { useState } from 'react';
import bitefinderLogo from './bitefinder-logo.png';
import { AppView } from '../types';

interface HeaderProps {
  onNavigate?: (view: AppView) => void;
}

const Header: React.FC<HeaderProps> = ({ onNavigate }) => {
  const [showGuidesDropdown, setShowGuidesDropdown] = useState(false);

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

        <div className="flex items-center gap-6">
          {/* Navigation - only show if onNavigate is provided */}
          {onNavigate && (
            <>
              {/* About Link */}
              <button
                onClick={() => onNavigate(AppView.ABOUT)}
                className="text-forest-200 hover:text-earth-100 transition-colors font-medium text-sm"
              >
                About
              </button>

              {/* Guides Dropdown */}
              <div className="relative">
                <button
                  onClick={() => setShowGuidesDropdown(!showGuidesDropdown)}
                  onBlur={() => setTimeout(() => setShowGuidesDropdown(false), 200)}
                  className="flex items-center gap-1 text-forest-200 hover:text-earth-100 transition-colors font-medium text-sm"
                >
                  <span>Guides</span>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2}
                    stroke="currentColor"
                    className={`w-3 h-3 transition-transform ${showGuidesDropdown ? 'rotate-180' : ''}`}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
                  </svg>
                </button>

                {/* Dropdown Menu */}
                {showGuidesDropdown && (
                  <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-xl border-2 border-earth-200 py-2 z-50">
                    <button
                      onClick={() => {
                        onNavigate(AppView.PREVENTION_GUIDE);
                        setShowGuidesDropdown(false);
                      }}
                      className="w-full text-left px-4 py-3 text-earth-900 hover:bg-forest-50 transition-colors flex items-center gap-3"
                    >
                      <span className="text-2xl">🛡️</span>
                      <div>
                        <div className="font-medium">Prevention Guide</div>
                        <div className="text-xs text-earth-600">Avoid bites before they happen</div>
                      </div>
                    </button>

                    <button
                      onClick={() => {
                        onNavigate(AppView.SEASONAL_CALENDAR);
                        setShowGuidesDropdown(false);
                      }}
                      className="w-full text-left px-4 py-3 text-earth-900 hover:bg-forest-50 transition-colors flex items-center gap-3"
                    >
                      <span className="text-2xl">📅</span>
                      <div>
                        <div className="font-medium">Seasonal Calendar</div>
                        <div className="text-xs text-earth-600">What's active when & where</div>
                      </div>
                    </button>

                    <button
                      onClick={() => {
                        onNavigate(AppView.BUG_EDUCATION);
                        setShowGuidesDropdown(false);
                      }}
                      className="w-full text-left px-4 py-3 text-earth-900 hover:bg-forest-50 transition-colors flex items-center gap-3"
                    >
                      <span className="text-2xl">🐛</span>
                      <div>
                        <div className="font-medium">Bug Education</div>
                        <div className="text-xs text-earth-600">Learn about common biters</div>
                      </div>
                    </button>
                  </div>
                )}
              </div>

              {/* Identify Bite Button (renamed from Home) */}
              <button
                onClick={() => onNavigate(AppView.HOME)}
                className="text-sm font-medium bg-forest-800 px-3 py-1 rounded-full text-forest-200 border border-forest-600 hover:bg-forest-700 transition-colors"
              >
                🔍 Identify Bug
              </button>
            </>
          )}

          {/* Version badge - only show when no navigation */}
          {!onNavigate && (
            <div className="hidden sm:block">
              <span className="text-sm font-medium bg-forest-800 px-3 py-1 rounded-full text-forest-200 border border-forest-600">
                v1.0 • Ranger Rick AI
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
