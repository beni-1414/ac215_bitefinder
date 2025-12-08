import React, { useState } from 'react';

interface SeasonalBugCalendarProps {
  onBack: () => void;
}

const SeasonalBugCalendar: React.FC<SeasonalBugCalendarProps> = ({ onBack }) => {
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<string>('all');

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  const regions = [
    { id: 'all', name: 'All Regions', icon: '🌎' },
    { id: 'northeast', name: 'Northeast', icon: '🍁' },
    { id: 'southeast', name: 'Southeast', icon: '🌴' },
    { id: 'midwest', name: 'Midwest', icon: '🌾' },
    { id: 'southwest', name: 'Southwest', icon: '🌵' },
    { id: 'west', name: 'West Coast', icon: '🌊' }
  ];

  const bugData: Record<string, any> = {
    'January': {
      all: { bugs: ['Fleas (indoors)', 'Bed bugs'], activity: 'Low', note: 'Most bugs dormant in cold climates' },
      northeast: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Very Low', note: 'Indoor pests only' },
      southeast: { bugs: ['Fleas', 'Bed bugs', 'Fire ants'], activity: 'Low-Moderate', note: 'Milder weather keeps some active' },
      midwest: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Very Low', note: 'Cold temperatures' },
      southwest: { bugs: ['Fleas', 'Fire ants'], activity: 'Low', note: 'Cooler but some activity' },
      west: { bugs: ['Fleas', 'Bed bugs'], activity: 'Low', note: 'Indoor pests primary concern' }
    },
    'February': {
      all: { bugs: ['Fleas (indoors)', 'Bed bugs'], activity: 'Low', note: 'Still winter dormancy' },
      northeast: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Very Low', note: 'Peak indoor pest season' },
      southeast: { bugs: ['Fleas', 'Fire ants', 'Mosquitoes (coastal)'], activity: 'Moderate', note: 'Early spring activity begins' },
      midwest: { bugs: ['Bed bugs'], activity: 'Very Low', note: 'Still very cold' },
      southwest: { bugs: ['Fleas', 'Fire ants', 'Spiders'], activity: 'Low-Moderate', note: 'Warming up' },
      west: { bugs: ['Fleas', 'Bed bugs'], activity: 'Low', note: 'Rainy season limits outdoor pests' }
    },
    'March': {
      all: { bugs: ['Ticks (emerging)', 'Fleas', 'Ants'], activity: 'Moderate', note: 'Spring awakening begins' },
      northeast: { bugs: ['Ticks (early)', 'Bed bugs', 'Ants'], activity: 'Low-Moderate', note: 'Tick season starts' },
      southeast: { bugs: ['Mosquitoes', 'Ticks', 'Fire ants', 'Fleas'], activity: 'Moderate-High', note: 'Full spring activity' },
      midwest: { bugs: ['Ticks', 'Ants', 'Fleas'], activity: 'Moderate', note: 'Thawing brings bugs out' },
      southwest: { bugs: ['Fire ants', 'Spiders', 'Fleas'], activity: 'Moderate', note: 'Desert bugs active' },
      west: { bugs: ['Ticks', 'Ants', 'Fleas'], activity: 'Moderate', note: 'Spring rains bring bugs' }
    },
    'April': {
      all: { bugs: ['Ticks', 'Mosquitoes', 'Fleas', 'Ants'], activity: 'High', note: 'Peak spring season begins' },
      northeast: { bugs: ['Ticks', 'Mosquitoes', 'Ants', 'Fleas'], activity: 'Moderate-High', note: 'Tick activity increases' },
      southeast: { bugs: ['Mosquitoes', 'Ticks', 'Fire ants', 'Chiggers', 'Fleas'], activity: 'High', note: 'Peak season begins' },
      midwest: { bugs: ['Ticks', 'Mosquitoes', 'Ants'], activity: 'Moderate-High', note: 'Spring fully underway' },
      southwest: { bugs: ['Fire ants', 'Spiders', 'Mosquitoes'], activity: 'Moderate-High', note: 'Before extreme heat' },
      west: { bugs: ['Ticks', 'Mosquitoes', 'Ants'], activity: 'Moderate-High', note: 'Wildflower season brings bugs' }
    },
    'May': {
      all: { bugs: ['Ticks', 'Mosquitoes', 'Chiggers', 'Fleas', 'Bed bugs'], activity: 'Very High', note: 'Peak bug season nationwide' },
      northeast: { bugs: ['Ticks', 'Mosquitoes', 'Black flies', 'Fleas'], activity: 'High', note: 'Lyme disease peak risk' },
      southeast: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Fire ants', 'Fleas'], activity: 'Very High', note: 'All pests very active' },
      midwest: { bugs: ['Ticks', 'Mosquitoes', 'Chiggers', 'Fleas'], activity: 'High', note: 'Humid weather ideal for bugs' },
      southwest: { bugs: ['Fire ants', 'Spiders', 'Mosquitoes'], activity: 'High', note: 'Before summer heat peak' },
      west: { bugs: ['Ticks', 'Mosquitoes', 'Fleas'], activity: 'High', note: 'Hiking season = tick season' }
    },
    'June': {
      all: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Fleas', 'Bed bugs'], activity: 'Very High', note: 'Summer peak season' },
      northeast: { bugs: ['Mosquitoes', 'Ticks', 'Black flies', 'Deer flies'], activity: 'Very High', note: 'Peak outdoor pest season' },
      southeast: { bugs: ['Mosquitoes', 'Chiggers', 'Ticks', 'Fire ants', 'Bed bugs'], activity: 'Very High', note: 'Hot & humid = bug heaven' },
      midwest: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Fleas'], activity: 'Very High', note: 'Summer camping caution needed' },
      southwest: { bugs: ['Mosquitoes', 'Spiders', 'Fire ants'], activity: 'High', note: 'Monsoon season approaching' },
      west: { bugs: ['Mosquitoes', 'Ticks', 'Fleas'], activity: 'High', note: 'Dry season, some activity' }
    },
    'July': {
      all: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Fleas', 'Fire ants'], activity: 'Very High', note: 'Peak summer activity' },
      northeast: { bugs: ['Mosquitoes', 'Ticks', 'Deer flies', 'Horse flies'], activity: 'Very High', note: 'Swampy areas worst' },
      southeast: { bugs: ['Mosquitoes', 'Chiggers', 'Fire ants', 'Bed bugs'], activity: 'Very High', note: 'Extreme heat & humidity' },
      midwest: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Harvest mites'], activity: 'Very High', note: 'Fields and woods active' },
      southwest: { bugs: ['Mosquitoes (monsoon)', 'Spiders', 'Fire ants'], activity: 'Very High', note: 'Monsoon brings mosquitoes' },
      west: { bugs: ['Mosquitoes', 'Ticks', 'Spiders'], activity: 'Moderate-High', note: 'Coastal areas cooler' }
    },
    'August': {
      all: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Fleas', 'Wasps'], activity: 'Very High', note: 'Late summer peak' },
      northeast: { bugs: ['Mosquitoes', 'Ticks', 'Wasps', 'Yellow jackets'], activity: 'Very High', note: 'Wasps become aggressive' },
      southeast: { bugs: ['Mosquitoes', 'Chiggers', 'Fire ants', 'Fleas'], activity: 'Very High', note: 'Still very active' },
      midwest: { bugs: ['Mosquitoes', 'Ticks', 'Chiggers', 'Wasps'], activity: 'High', note: 'Harvest season begins' },
      southwest: { bugs: ['Mosquitoes', 'Spiders', 'Fire ants'], activity: 'High', note: 'Monsoon continues' },
      west: { bugs: ['Mosquitoes', 'Wasps', 'Ticks'], activity: 'Moderate-High', note: 'Fire season - bugs scattered' }
    },
    'September': {
      all: { bugs: ['Mosquitoes', 'Ticks', 'Fleas', 'Wasps'], activity: 'High', note: 'Fall activity still strong' },
      northeast: { bugs: ['Mosquitoes', 'Ticks', 'Wasps', 'Fleas'], activity: 'Moderate-High', note: 'Cooling but still active' },
      southeast: { bugs: ['Mosquitoes', 'Ticks', 'Fire ants', 'Fleas'], activity: 'High', note: 'Still warm and active' },
      midwest: { bugs: ['Mosquitoes', 'Ticks', 'Fleas'], activity: 'Moderate-High', note: 'Harvest moon brings bugs' },
      southwest: { bugs: ['Mosquitoes', 'Spiders', 'Fire ants'], activity: 'Moderate', note: 'Cooling down' },
      west: { bugs: ['Mosquitoes', 'Ticks', 'Fleas'], activity: 'Moderate', note: 'Fall rains bring activity' }
    },
    'October': {
      all: { bugs: ['Ticks', 'Fleas', 'Bed bugs'], activity: 'Moderate', note: 'Fall season, decreasing activity' },
      northeast: { bugs: ['Ticks', 'Fleas', 'Bed bugs'], activity: 'Moderate', note: 'Second tick season' },
      southeast: { bugs: ['Mosquitoes', 'Ticks', 'Fire ants', 'Fleas'], activity: 'Moderate-High', note: 'Still active in warm weather' },
      midwest: { bugs: ['Ticks', 'Fleas', 'Bed bugs'], activity: 'Moderate', note: 'Seeking indoor shelter' },
      southwest: { bugs: ['Fire ants', 'Spiders', 'Fleas'], activity: 'Moderate', note: 'Pleasant weather brings activity' },
      west: { bugs: ['Fleas', 'Bed bugs', 'Ticks'], activity: 'Moderate', note: 'Rainy season begins' }
    },
    'November': {
      all: { bugs: ['Fleas', 'Bed bugs'], activity: 'Low-Moderate', note: 'Indoor pests become concern' },
      northeast: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Low', note: 'Most outdoor bugs dormant' },
      southeast: { bugs: ['Fleas', 'Fire ants', 'Bed bugs'], activity: 'Moderate', note: 'Milder weather extends season' },
      midwest: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Low', note: 'Cold weather arriving' },
      southwest: { bugs: ['Fleas', 'Fire ants'], activity: 'Low-Moderate', note: 'Pleasant outdoor weather' },
      west: { bugs: ['Fleas', 'Bed bugs'], activity: 'Low', note: 'Rainy season' }
    },
    'December': {
      all: { bugs: ['Fleas (indoors)', 'Bed bugs'], activity: 'Low', note: 'Winter dormancy' },
      northeast: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Very Low', note: 'Indoor pests only' },
      southeast: { bugs: ['Fleas', 'Bed bugs', 'Fire ants'], activity: 'Low-Moderate', note: 'Some activity in warm spells' },
      midwest: { bugs: ['Bed bugs', 'Fleas (indoors)'], activity: 'Very Low', note: 'Winter conditions' },
      southwest: { bugs: ['Fleas', 'Fire ants'], activity: 'Low', note: 'Cool but some activity' },
      west: { bugs: ['Fleas', 'Bed bugs'], activity: 'Low', note: 'Rainy winter season' }
    }
  };

  const getActivityColor = (activity: string) => {
    if (activity.includes('Very High')) return 'bg-red-100 text-red-800 border-red-300';
    if (activity.includes('High')) return 'bg-orange-100 text-orange-800 border-orange-300';
    if (activity.includes('Moderate')) return 'bg-amber-100 text-amber-800 border-amber-300';
    return 'bg-green-100 text-green-800 border-green-300';
  };

  const selectedMonthData = selectedMonth ? bugData[selectedMonth][selectedRegion] : null;

  return (
    <div className="min-h-screen bg-earth-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
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
            <div className="text-7xl">📅</div>
            <div>
              <h1 className="text-4xl font-serif font-bold mb-2">
                Seasonal Bug Calendar
              </h1>
              <p className="text-xl text-forest-200">
                Know what's biting when and where
              </p>
            </div>
          </div>
        </div>

        {/* Region Filter */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border-2 border-earth-200">
          <h3 className="text-xl font-serif font-bold text-earth-900 mb-4">Select Your U.S. Region</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {regions.map((region) => (
              <button
                key={region.id}
                onClick={() => setSelectedRegion(region.id)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedRegion === region.id
                    ? 'border-forest-600 bg-forest-50 shadow-md'
                    : 'border-earth-200 hover:border-forest-400'
                }`}
              >
                <div className="text-3xl mb-1">{region.icon}</div>
                <div className="text-sm font-medium text-earth-900">{region.name}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Month Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-8">
          {months.map((month) => {
            const data = bugData[month][selectedRegion];
            return (
              <button
                key={month}
                onClick={() => setSelectedMonth(month)}
                className={`p-6 rounded-xl border-2 transition-all text-left ${
                  selectedMonth === month
                    ? 'border-forest-600 bg-forest-50 shadow-lg'
                    : 'border-earth-200 bg-white hover:border-forest-400 hover:shadow-md'
                }`}
              >
                <div className="font-bold text-lg text-earth-900 mb-2">{month}</div>
                <div className={`text-xs px-3 py-1 rounded-full inline-block border ${getActivityColor(data.activity)}`}>
                  {data.activity}
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Month Detail */}
        {selectedMonthData && selectedMonth && (
          <div className="bg-white rounded-xl shadow-2xl p-8 border-2 border-earth-200">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-3xl font-serif font-bold text-earth-900">
                {selectedMonth} - {regions.find(r => r.id === selectedRegion)?.name}
              </h2>
              <div className={`px-6 py-2 rounded-full text-sm font-bold border-2 ${getActivityColor(selectedMonthData.activity)}`}>
                {selectedMonthData.activity} Activity
              </div>
            </div>

            <div className="space-y-6">
              {/* Common Bugs */}
              <div>
                <h3 className="text-xl font-bold text-earth-900 mb-3 flex items-center gap-2">
                  <span className="text-2xl">🐛</span>
                  Active Bugs This Month
                </h3>
                <div className="flex flex-wrap gap-3">
                  {selectedMonthData.bugs.map((bug: string, idx: number) => (
                    <span key={idx} className="bg-forest-100 text-forest-800 px-4 py-2 rounded-full text-sm font-medium border border-forest-300">
                      {bug}
                    </span>
                  ))}
                </div>
              </div>

              {/* Ranger's Notes */}
              <div className="bg-forest-50 rounded-lg p-6 border-2 border-forest-200">
                <h3 className="font-bold text-forest-900 mb-2 flex items-center gap-2 text-lg">
                  <span className="text-2xl">🎒</span>
                  Ranger Rick's Seasonal Notes
                </h3>
                <p className="text-forest-800 text-lg">{selectedMonthData.note}</p>
              </div>
            </div>
          </div>
        )}

        {!selectedMonth && (
          <div className="bg-forest-50 rounded-xl p-12 text-center border-2 border-forest-200">
            <h3 className="text-2xl font-serif font-bold text-earth-900 mb-2">
              Select a Month
            </h3>
            <p className="text-earth-600 text-lg">
              Click any month above to see what bugs are active in your region
            </p>
          </div>
        )}

        {/* Pro Tips */}
        <div className="mt-12 bg-gradient-to-r from-forest-600 to-forest-800 rounded-xl shadow-2xl p-8 text-white border-4 border-forest-900">
          <h2 className="text-2xl font-serif font-bold mb-4 flex items-center gap-2">
            <span className="text-3xl">💡</span>
            Ranger Rick's Seasonal Tips
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-forest-700 rounded-lg p-4">
              <h3 className="font-bold mb-2">🌸 Spring (Mar-May)</h3>
              <p className="text-forest-100 text-sm">Tick season begins! Check yourself after every outdoor activity. Ticks are most active in spring and fall.</p>
            </div>
            <div className="bg-forest-700 rounded-lg p-4">
              <h3 className="font-bold mb-2">☀️ Summer (Jun-Aug)</h3>
              <p className="text-forest-100 text-sm">Peak mosquito season. Use repellent at dawn and dusk. Stay hydrated and watch for heat exhaustion too!</p>
            </div>
            <div className="bg-forest-700 rounded-lg p-4">
              <h3 className="font-bold mb-2">🍂 Fall (Sep-Nov)</h3>
              <p className="text-forest-100 text-sm">Second tick season! Bugs seek warmth indoors. Check for bed bugs and fleas as pests move inside.</p>
            </div>
            <div className="bg-forest-700 rounded-lg p-4">
              <h3 className="font-bold mb-2">❄️ Winter (Dec-Feb)</h3>
              <p className="text-forest-100 text-sm">Indoor pest season. Southern states still have outdoor activity. Watch for bed bugs in hotels during holiday travel.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SeasonalBugCalendar;
