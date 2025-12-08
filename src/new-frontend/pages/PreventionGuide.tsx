import React from 'react';

interface PreventionGuideProps {
  onBack: () => void;
}

const PreventionGuide: React.FC<PreventionGuideProps> = ({ onBack }) => {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center text-earth-600 hover:text-forest-700 transition-colors font-medium mb-6 text-lg"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4 mr-1">
          <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
        </svg>
        Back to Home
      </button>

      {/* Header */}
      <div className="bg-gradient-to-r from-forest-600 to-forest-800 rounded-xl shadow-xl p-8 mb-8 border-4 border-forest-900 text-white">
        <div className="flex items-center gap-4">
          <div className="text-7xl">🛡️</div>
          <div>
            <h1 className="text-4xl font-serif font-bold mb-2">
              Bite Prevention Guide
            </h1>
            <p className="text-xl text-forest-200">
              Stay safe on the trail with Ranger Rick's proven tips
            </p>
          </div>
        </div>
      </div>

      {/* Content Sections */}
      <div className="space-y-6">

        {/* Before You Go */}
        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-earth-200">
          <h2 className="text-2xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
            <span className="text-3xl">🎒</span>
            Before You Go Outside
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-forest-100 rounded-full flex items-center justify-center text-2xl">
                🧴
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Apply Insect Repellent</h3>
                <p className="text-earth-700">
                  Use EPA-registered repellents with 20-30% DEET, picaridin, or oil of lemon eucalyptus.
                  Apply to exposed skin and clothing. Reapply every 2-4 hours or after swimming.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-forest-100 rounded-full flex items-center justify-center text-2xl">
                👕
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Dress Appropriately</h3>
                <p className="text-earth-700">
                  Wear light-colored, long-sleeved shirts and long pants. Tuck pants into socks for extra
                  protection. Consider treating clothes with permethrin for added defense.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-forest-100 rounded-full flex items-center justify-center text-2xl">
                ⏰
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Time Your Activities</h3>
                <p className="text-earth-700">
                  Mosquitoes are most active at dawn and dusk. If possible, plan outdoor activities
                  for mid-day when bug activity is lower.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* While You're Outside */}
        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-earth-200">
          <h2 className="text-2xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
            <span className="text-3xl">🌲</span>
            While You're Outside
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center text-2xl">
                🚫
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Avoid Bug Hotspots</h3>
                <p className="text-earth-700">
                  Stay away from standing water, tall grass, leaf piles, and dense brush.
                  Walk in the center of trails to minimize contact with vegetation.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center text-2xl">
                🏕️
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Set Up Camp Smart</h3>
                <p className="text-earth-700">
                  Choose dry, elevated areas away from standing water. Use mosquito netting on tents
                  and sleeping areas. Keep food sealed to avoid attracting insects.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center text-2xl">
                👀
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Stay Vigilant</h3>
                <p className="text-earth-700">
                  Check yourself periodically for ticks and other insects. Brush off any bugs you see
                  before they have a chance to bite.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* After You Return */}
        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-earth-200">
          <h2 className="text-2xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
            <span className="text-3xl">🏠</span>
            After You Return
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">
                🔍
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Do a Thorough Tick Check</h3>
                <p className="text-earth-700">
                  Check your entire body, paying special attention to armpits, groin, scalp, behind
                  ears, and behind knees. Check children and pets too.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">
                🚿
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Shower Within 2 Hours</h3>
                <p className="text-earth-700">
                  Showering soon after coming indoors can help wash off unattached ticks and makes
                  it easier to spot any that may have attached.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">
                👔
              </div>
              <div>
                <h3 className="font-bold text-earth-900 mb-1">Wash Your Clothes</h3>
                <p className="text-earth-700">
                  Tumble dry clothes on high heat for 10 minutes to kill any remaining ticks.
                  If clothes need washing first, use hot water.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Specific Scenarios */}
        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-earth-200">
          <h2 className="text-2xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
            <span className="text-3xl">🎯</span>
            Specific Scenarios
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-earth-50 rounded-lg p-4 border border-earth-200">
              <h3 className="font-bold text-earth-900 mb-2 flex items-center gap-2">
                <span className="text-xl">⛺</span>
                Camping
              </h3>
              <ul className="text-sm text-earth-700 space-y-1">
                <li>• Use treated mosquito netting</li>
                <li>• Keep tent zipped at all times</li>
                <li>• Avoid camping near water</li>
                <li>• Hang food away from sleeping area</li>
              </ul>
            </div>

            <div className="bg-earth-50 rounded-lg p-4 border border-earth-200">
              <h3 className="font-bold text-earth-900 mb-2 flex items-center gap-2">
                <span className="text-xl">🥾</span>
                Hiking
              </h3>
              <ul className="text-sm text-earth-700 space-y-1">
                <li>• Stay on cleared trails</li>
                <li>• Tuck pants into socks/boots</li>
                <li>• Wear light colors to spot bugs</li>
                <li>• Take breaks in sunny, open areas</li>
              </ul>
            </div>

            <div className="bg-earth-50 rounded-lg p-4 border border-earth-200">
              <h3 className="font-bold text-earth-900 mb-2 flex items-center gap-2">
                <span className="text-xl">🌻</span>
                Gardening
              </h3>
              <ul className="text-sm text-earth-700 space-y-1">
                <li>• Wear gloves and closed-toe shoes</li>
                <li>• Apply repellent before starting</li>
                <li>• Treat garden clothes with permethrin</li>
                <li>• Clear leaf litter and tall grass</li>
              </ul>
            </div>

            <div className="bg-earth-50 rounded-lg p-4 border border-earth-200">
              <h3 className="font-bold text-earth-900 mb-2 flex items-center gap-2">
                <span className="text-xl">🏡</span>
                Backyard
              </h3>
              <ul className="text-sm text-earth-700 space-y-1">
                <li>• Empty standing water weekly</li>
                <li>• Keep grass short and trimmed</li>
                <li>• Create a wood chip/gravel barrier</li>
                <li>• Use citronella candles for gatherings</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Ranger Rick's Pro Tips */}
        <div className="bg-gradient-to-r from-forest-600 to-forest-800 rounded-xl shadow-xl p-6 text-white border-4 border-forest-900">
          <h2 className="text-2xl font-serif font-bold mb-4 flex items-center gap-2">
            <span className="text-3xl">💡</span>
            Ranger Rick's Pro Tips
          </h2>
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="text-xl mt-1">✓</span>
              <span className="text-forest-50">
                <strong>Natural repellents work</strong> - but need more frequent application than DEET-based products
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-xl mt-1">✓</span>
              <span className="text-forest-50">
                <strong>Permethrin-treated clothing</strong> can remain effective through multiple washings
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-xl mt-1">✓</span>
              <span className="text-forest-50">
                <strong>Ticks can't jump or fly</strong> - they grab onto you when you brush past vegetation
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-xl mt-1">✓</span>
              <span className="text-forest-50">
                <strong>Check your pets</strong> - they can bring ticks into your home
              </span>
            </li>
          </ul>
        </div>

      </div>
    </div>
  );
};

export default PreventionGuide;
