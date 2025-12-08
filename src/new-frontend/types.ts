// App view states
export enum AppView {
  HOME = 'HOME',
  ANALYZING = 'ANALYZING',
  RESULT = 'RESULT',
  PREVENTION_GUIDE = 'PREVENTION_GUIDE',
  SEASONAL_CALENDAR = 'SEASONAL_CALENDAR',
  BUG_EDUCATION = 'BUG_EDUCATION',
  ABOUT = 'ABOUT'
}

// Bite analysis result
export interface BiteAnalysis {
  bugName: string;
  scientificName: string;
  description: string;
  dangerLevel: 'Low' | 'Moderate' | 'High' | 'Emergency';
}

// Chat message
export interface ChatMessage {
  id: string;
  sender: 'user' | 'ranger';
  text: string;
  timestamp: Date;
}
