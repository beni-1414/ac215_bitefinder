import React from 'react';

interface AboutPageProps {
  onBack: () => void;
}

const AboutPage: React.FC<AboutPageProps> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-earth-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Back button */}
        <button
          onClick={onBack}
          className="flex items-center text-earth-600 hover:text-forest-700 transition-colors font-medium mb-6 text-lg"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5 mr-2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
          Back to Home
        </button>



      {/* Header */}
              <div className="bg-gradient-to-r from-forest-600 to-forest-800 rounded-xl shadow-xl p-8 mb-8 border-4 border-forest-900 text-white">
                <div className="flex items-center gap-4">
                  <div className="text-7xl">🔍</div>
                  <div>
                    <h1 className="text-4xl font-serif font-bold mb-2">
                      About BiteFinder
                    </h1>
                    <p className="text-lg mb-2">
                      Your AI-powered wilderness companion
                    </p>
                  </div>
                </div>
              </div>
        {/* Content Sections */}
        <div className="space-y-8">

          {/* What is BiteFinder */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-2 border-earth-200">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
              <span className="text-4xl">🪲</span>
              What is BiteFinder?
            </h2>
            <p className="text-lg text-earth-700 leading-relaxed mb-4">
              BiteFinder is an AI-powered tool designed to help you identify insect bites and get personalized advice
              from Ranger Rick, your virtual wilderness expert. Whether you're camping in the backcountry, hiking
              through forests, or just enjoying your backyard, BiteFinder is here to help you stay safe and informed.
            </p>
            <p className="text-lg text-earth-700 leading-relaxed">
              Simply upload a photo of your bite, describe your symptoms and where you were bitten, and our AI will
              analyze the image to identify the likely culprit. Then, chat with Ranger Rick to get tailored advice
              about treatment, prevention, and when to seek medical attention.
            </p>
          </div>

          {/* How It Works */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-2 border-earth-200">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-6 flex items-center gap-2">
              <span className="text-4xl">⚙️</span>
              How It Works
            </h2>
            <div className="space-y-6">
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-forest-600 text-white rounded-full flex items-center justify-center font-bold text-xl">
                  1
                </div>
                <div>
                  <h3 className="text-xl font-bold text-earth-900 mb-2">Upload & Describe</h3>
                  <p className="text-earth-700">
                    Take a clear photo of your bite and provide details about your symptoms and location.
                    The more information you provide, the better our AI can help!
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-forest-600 text-white rounded-full flex items-center justify-center font-bold text-xl">
                  2
                </div>
                <div>
                  <h3 className="text-xl font-bold text-earth-900 mb-2">AI Analysis</h3>
                  <p className="text-earth-700">
                    Our advanced AI model analyzes your photo and description to identify the type of insect
                    that likely caused the bite, along with a confidence level and risk assessment.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-forest-600 text-white rounded-full flex items-center justify-center font-bold text-xl">
                  3
                </div>
                <div>
                  <h3 className="text-xl font-bold text-earth-900 mb-2">Chat with Ranger Rick</h3>
                  <p className="text-earth-700">
                    Get personalized advice from Ranger Rick about treatment options, prevention tips,
                    and information about the insect. Ask follow-up questions anytime!
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Meet Ranger Rick */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-2 border-earth-200">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
              <span className="text-4xl">🎒</span>
              Meet Ranger Rick
            </h2>
            <div className="bg-forest-50 rounded-lg p-6 border-2 border-forest-200">
              <p className="text-lg text-forest-900 leading-relaxed mb-4">
                Ranger Rick is your friendly AI wilderness expert, trained on extensive knowledge about insects,
                bug bites, outdoor safety, and first aid. He's here to answer your questions in a friendly,
                approachable way - just like talking to an experienced park ranger on the trail.
              </p>
              <p className="text-lg text-forest-900 leading-relaxed">
                Whether you need advice on treating a bite, preventing future encounters, or just want to learn
                more about the critter that got you, Ranger Rick has you covered with reliable, helpful information.
              </p>
            </div>
          </div>

          {/* Technology */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-2 border-earth-200">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
              <span className="text-4xl">🤖</span>
              The Technology
            </h2>
            <p className="text-lg text-earth-700 leading-relaxed mb-4">
              BiteFinder uses state-of-the-art computer vision and natural language processing to analyze
              your bite photos and provide helpful guidance. Our AI has been trained on thousands of insect
              bite images and extensive medical and entomological literature.
            </p>
            <div className="bg-amber-50 rounded-lg p-4 border-2 border-amber-200">
              <p className="text-earth-800">
                <strong>Built with:</strong> React, TypeScript, Tailwind CSS, CLIP vision‑language models for image recognition, plus large language models for conversational AI.
              </p>
            </div>
          </div>

          {/* Important Disclaimer */}
          <div className="bg-orange-50 rounded-xl shadow-lg p-8 border-2 border-orange-200">
            <h2 className="text-2xl font-serif font-bold text-orange-900 mb-4 flex items-center gap-2">
              <span className="text-3xl">⚠️</span>
              Important Medical Disclaimer
            </h2>
            <div className="space-y-3 text-orange-900">
              <p className="text-lg font-medium">
                BiteFinder is an educational tool and should NOT be used as a substitute for professional medical advice,
                diagnosis, or treatment.
              </p>
              <ul className="list-disc list-inside space-y-2 text-orange-800">
                <li>Always consult with a qualified healthcare provider for medical concerns</li>
                <li>Seek immediate medical attention for severe reactions, difficulty breathing, or signs of infection</li>
                <li>If you suspect a venomous bite (spider, snake, etc.), call emergency services immediately</li>
                <li>AI predictions are not 100% accurate—use your best judgment and seek professional help when in doubt</li>
              </ul>
            </div>
          </div>

          {/* Our Mission */}
          <div className="bg-gradient-to-r from-forest-100 to-earth-100 rounded-xl shadow-lg p-8 border-2 border-forest-200">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
              <span className="text-4xl">🎯</span>
              Our Mission
            </h2>
            <p className="text-lg text-forest-900 leading-relaxed mb-4">
              We believe that everyone should have access to reliable information about insect bites and outdoor safety.
              Our mission is to empower outdoor enthusiasts, families, and adventurers with the knowledge they need to
              enjoy nature safely and confidently.
            </p>
            <p className="text-lg text-forest-900 leading-relaxed">
              By combining AI technology with expert wilderness knowledge, we're making it easier than ever to identify
              bug bites, understand the risks, and take appropriate action - whether you're deep in the wilderness or
              just steps from your back door.
            </p>
          </div>

          {/* Get Started CTA */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-2 border-earth-200 text-center">
            <h2 className="text-3xl font-serif font-bold text-forest-800 mb-4">
              Ready to Identify Your Bite?
            </h2>
            <p className="text-lg text-earth-700 mb-6">
              Upload a photo and let Ranger Rick help you figure out what bit you!
            </p>
            <button
              onClick={onBack}
              className="bg-forest-600 hover:bg-forest-700 text-white font-bold py-4 px-8 rounded-lg text-lg transition-colors inline-flex items-center gap-2"
            >
              <span>Get Started</span>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
              </svg>
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default AboutPage;
