export interface BiteAnalysis {
  bugName: string;
  scientificName: string;
  description: string;
  dangerLevel: 'Low' | 'Moderate' | 'High' | 'Emergency';
  symptoms: string[];
  treatmentTips: string[];
  preventionTips: string[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ranger';
  text: string;
  timestamp: Date;
}

export enum AppView {
  HOME = 'HOME',
  ANALYZING = 'ANALYZING',
  RESULT = 'RESULT',
}

export interface AnalysisRequest {
  imageBase64: string;
  userNotes: string;
}
