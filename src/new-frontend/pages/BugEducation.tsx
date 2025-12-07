import React, { useState } from 'react';

interface BugEducationProps {
  onBack: () => void;
}

const BugEducation: React.FC<BugEducationProps> = ({ onBack }) => {
  const [selectedBug, setSelectedBug] = useState<string | null>(null);

  const bugs = [
    {
      id: 'mosquitoes',
      name: 'Mosquitoes',
      emoji: '🦟',
      scientific: 'Culicidae family',
      danger: 'Moderate-High',
      description: 'Small flying insects that feed on blood. Only females bite.',
      identification: [
        'Thin, long legs and body',
        'Long proboscis (needle-like mouthpart)',
        'Size: 3-6mm long',
        'High-pitched buzzing sound'
      ],
      biteAppearance: 'Small, raised, itchy bumps that appear shortly after the bite. Usually red or pink with a central dot.',
      diseases: [
        'West Nile Virus',
        'Zika Virus',
        'Dengue Fever',
        'Malaria (in tropical regions)',
        'Eastern Equine Encephalitis'
      ],
      habitat: 'Standing water, marshes, ponds, birdbaths, gutters, anywhere water collects',
      season: 'Late spring through fall (year-round in warm climates)',
      prevention: [
        'Eliminate standing water around your home',
        'Use EPA-registered insect repellent',
        'Wear long sleeves and pants at dawn/dusk',
        'Install or repair window screens',
        'Use mosquito netting when camping'
      ],
      treatment: 'Clean with soap and water. Apply anti-itch cream or calamine lotion. Take antihistamines for severe itching.'
    },
    {
      id: 'ticks',
      name: 'Ticks',
      emoji: '🕷️',
      scientific: 'Ixodida order',
      danger: 'High',
      description: 'Small arachnids that attach to skin and feed on blood for days.',
      identification: [
        'Oval-shaped, flat body (when unfed)',
        'Eight legs (they\'re arachnids, not insects!)',
        'Size: 1-5mm unfed, up to 10mm when engorged',
        'Brown, black, or reddish-brown color'
      ],
      biteAppearance: 'Often painless. May see the tick still attached. Red spot or rash at bite site. Bulls-eye rash indicates Lyme disease.',
      diseases: [
        'Lyme Disease (most common)',
        'Rocky Mountain Spotted Fever',
        'Anaplasmosis',
        'Babesiosis',
        'Powassan Virus',
        'Alpha-gal Syndrome (meat allergy)'
      ],
      habitat: 'Wooded areas, tall grass, leaf litter, brush piles, anywhere vegetation is thick',
      season: 'Spring and fall peak seasons, but active year-round in some areas',
      prevention: [
        'Walk in the center of trails',
        'Wear long pants tucked into socks',
        'Use permethrin on clothing',
        'Apply DEET or picaridin repellent',
        'Do full-body tick checks after being outdoors',
        'Shower within 2 hours of coming inside'
      ],
      treatment: 'Remove tick with fine-tipped tweezers (pull straight up). Clean area with rubbing alcohol. Save the tick in a sealed bag. Watch for symptoms for 30 days. See doctor if rash or fever develops.'
    },
    {
      id: 'bed-bugs',
      name: 'Bed Bugs',
      emoji: '🛏️',
      scientific: 'Cimex lectularius',
      danger: 'Low',
      description: 'Small parasitic insects that feed on human blood, typically at night.',
      identification: [
        'Flat, oval-shaped body (apple seed size)',
        'Reddish-brown color',
        'Size: 4-5mm long',
        'Cannot fly or jump',
        'Sweet, musty odor when infested'
      ],
      biteAppearance: 'Red, itchy welts often in lines or clusters (breakfast, lunch, dinner pattern). Bites appear hours to days after feeding.',
      diseases: [
        'No known disease transmission',
        'Can cause secondary infections from scratching',
        'Allergic reactions in some people',
        'Anxiety and sleep disturbances'
      ],
      habitat: 'Mattresses, box springs, bed frames, headboards, furniture, behind wallpaper, in electrical outlets',
      season: 'Year-round (indoor pest)',
      prevention: [
        'Inspect hotel rooms before unpacking',
        'Keep luggage off the floor when traveling',
        'Vacuum regularly and check mattress seams',
        'Use mattress and box spring encasements',
        'Avoid second-hand furniture',
        'Reduce clutter where they can hide'
      ],
      treatment: 'Wash bites with soap and water. Apply anti-itch cream. For infestation, contact a professional pest control service immediately.'
    },
    {
      id: 'fleas',
      name: 'Fleas',
      emoji: '🐕',
      scientific: 'Siphonaptera order',
      danger: 'Low-Moderate',
      description: 'Tiny jumping insects that feed on blood of mammals and birds.',
      identification: [
        'Very small (1-3mm)',
        'Dark brown or reddish-brown',
        'Flattened body side-to-side',
        'Powerful hind legs for jumping',
        'Can jump up to 8 inches high'
      ],
      biteAppearance: 'Small red bumps often with a red halo, extremely itchy. Common on ankles and legs. Often in groups of 2-3 bites.',
      diseases: [
        'Plague (rare but serious)',
        'Murine typhus',
        'Tapeworms (if flea is swallowed)',
        'Cat scratch disease (Bartonella)'
      ],
      habitat: 'On pets (dogs, cats), in carpets, bedding, upholstered furniture, cracks in floors',
      season: 'Peak in summer and fall, but year-round indoors with pets',
      prevention: [
        'Treat pets with vet-approved flea prevention',
        'Vacuum carpets and furniture frequently',
        'Wash pet bedding in hot water weekly',
        'Keep grass short in yard',
        'Use flea treatments in home if needed'
      ],
      treatment: 'Wash bites with antiseptic soap. Apply anti-itch cream or hydrocortisone. Treat home and pets to eliminate infestation.'
    },
    {
      id: 'chiggers',
      name: 'Chiggers',
      emoji: '🔴',
      scientific: 'Trombiculidae family',
      danger: 'Low',
      description: 'Tiny mite larvae that feed on skin cells, causing intense itching.',
      identification: [
        'Nearly microscopic (0.3mm)',
        'Bright red or orange color',
        'Not actually insects (they\'re mites)',
        'Only larvae bite humans',
        'Visible as tiny red dots on skin'
      ],
      biteAppearance: 'Intensely itchy red bumps or welts, often in areas where clothing is tight (waistband, sock line, armpits). Itching begins hours after exposure.',
      diseases: [
        'Scrub typhus (in certain regions)',
        'Generally no disease transmission in US',
        'Secondary infections from scratching'
      ],
      habitat: 'Tall grass, weeds, berry patches, woodland edges, damp areas with vegetation',
      season: 'Late spring through fall, peak in summer',
      prevention: [
        'Wear long pants and sleeves',
        'Tuck pants into boots',
        'Use DEET or permethrin repellent',
        'Avoid sitting directly on ground',
        'Shower immediately after outdoor activities',
        'Wash clothes in hot water'
      ],
      treatment: 'Shower with soap to remove larvae. Apply anti-itch cream or calamine lotion. Take oral antihistamines. Avoid scratching to prevent infection.'
    },
    {
      id: 'fire-ants',
      name: 'Fire Ants',
      emoji: '🐜',
      scientific: 'Solenopsis invicta',
      danger: 'Moderate-High',
      description: 'Aggressive ants that sting multiple times, injecting venom that causes burning pain.',
      identification: [
        'Reddish-brown color',
        'Size: 2-6mm long',
        'Build large dirt mounds (up to 18 inches)',
        'Swarm aggressively when disturbed',
        'Found mainly in southern US'
      ],
      biteAppearance: 'Painful, burning sensation immediately. White pustules develop within 24 hours. Multiple stings in circular pattern.',
      diseases: [
        'No disease transmission',
        'Can cause severe allergic reactions (anaphylaxis)',
        'Secondary infections from scratching',
        'Scarring from pustules'
      ],
      habitat: 'Soil mounds in yards, parks, fields, roadsides. Also invade homes, electrical boxes, and air conditioners',
      season: 'Most active in warm weather (spring through fall), but present year-round in southern states',
      prevention: [
        'Watch for mounds and avoid them',
        'Wear closed-toe shoes outdoors',
        'Don\'t sit or stand near mounds',
        'Treat mounds with approved pesticides',
        'Keep food sealed and clean up spills',
        'Seal home entry points'
      ],
      treatment: 'Move away quickly and brush off ants. Wash with soap and water. Apply ice to reduce swelling. Don\'t pop pustules. Seek emergency care if allergic reaction occurs.'
    },
    {
      id: 'spiders',
      name: 'Spiders',
      emoji: '🕷️',
      scientific: 'Araneae order',
      danger: 'Low-High (depends on species)',
      description: 'Eight-legged arachnids. Most are harmless, but some can deliver medically significant bites.',
      identification: [
        'Eight legs and two body segments',
        'Size varies widely (2mm to 50mm+)',
        'Most have eight eyes',
        'Build webs or hunt actively',
        'Venomous species: Black Widow, Brown Recluse'
      ],
      biteAppearance: 'Most bites: small red bump, minor swelling. Black Widow: severe pain, muscle cramps. Brown Recluse: bulls-eye pattern, tissue death possible.',
      diseases: [
        'No disease transmission',
        'Venom effects vary by species',
        'Black Widow: neurotoxic (affects nerves)',
        'Brown Recluse: necrotic (destroys tissue)',
        'Most spider bites are harmless'
      ],
      habitat: 'Everywhere! Webs in corners, woodpiles, sheds, garages, basements, under rocks, in vegetation',
      season: 'Year-round, but more visible in late summer and fall',
      prevention: [
        'Shake out shoes and clothing before wearing',
        'Wear gloves when handling firewood',
        'Keep storage areas clean and organized',
        'Seal cracks and gaps in home',
        'Remove webs regularly',
        'Keep beds away from walls'
      ],
      treatment: 'Most bites: clean with soap and water, apply ice, elevate if possible. Seek immediate medical care for Black Widow or Brown Recluse bites, or if severe symptoms develop.'
    },
    {
      id: 'ants',
      name: 'Ants (Common)',
      emoji: '🐜',
      scientific: 'Formicidae family',
      danger: 'Low',
      description: 'Social insects that live in colonies. Most species are harmless, but some can bite or sting.',
      identification: [
        'Three distinct body segments',
        'Elbowed antennae',
        'Size: 2-15mm depending on species',
        'Colors vary (black, red, brown)',
        'Live in organized colonies'
      ],
      biteAppearance: 'Small red bumps or welts. Mild pain and itching. Usually not serious unless allergic.',
      diseases: [
        'Generally no disease transmission',
        'Can contaminate food',
        'Rare allergic reactions possible'
      ],
      habitat: 'Soil, wood, under rocks, inside walls, around foundations, anywhere food is available',
      season: 'Most active in warm weather, but some species active year-round indoors',
      prevention: [
        'Keep food sealed and clean up crumbs',
        'Fix water leaks (ants need water)',
        'Seal entry points around home',
        'Trim vegetation away from house',
        'Store firewood away from home',
        'Use ant baits if needed'
      ],
      treatment: 'Wash bite area with soap and water. Apply ice if swollen. Use anti-itch cream if needed. Usually resolves on its own.'
    }
  ];

  const selectedBugData = bugs.find(b => b.id === selectedBug);

  const getDangerColor = (level: string) => {
    if (level === 'High' || level.includes('High')) return 'bg-red-100 text-red-800 border-red-300';
    if (level.includes('Moderate')) return 'bg-amber-100 text-amber-800 border-amber-300';
    return 'bg-green-100 text-green-800 border-green-300';
  };

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
            <div className="text-7xl">🐛</div>
            <div>
              <h1 className="text-4xl font-serif font-bold mb-2">
                Bug Education Center
              </h1>
              <p className="text-xl text-forest-200">
                Learn about common biting insects and how to stay safe
              </p>
            </div>
          </div>
        </div>

        {/* Bug Selection Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {bugs.map((bug) => (
            <button
              key={bug.id}
              onClick={() => setSelectedBug(bug.id)}
              className={`p-6 rounded-xl border-2 transition-all ${
                selectedBug === bug.id
                  ? 'border-forest-600 bg-forest-50 shadow-lg'
                  : 'border-earth-200 bg-white hover:border-forest-400 hover:shadow-md'
              }`}
            >
              <div className="text-5xl mb-3">{bug.emoji}</div>
              <div className="font-bold text-lg text-earth-900 mb-2">{bug.name}</div>
              <div className={`text-xs px-3 py-1 rounded-full inline-block border ${getDangerColor(bug.danger)}`}>
                {bug.danger}
              </div>
            </button>
          ))}
        </div>

        {/* Selected Bug Detail */}
        {selectedBugData && (
          <div className="bg-white rounded-xl shadow-2xl border-2 border-earth-200 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-forest-100 to-earth-100 p-8 border-b-2 border-earth-200">
              <div className="flex items-center gap-6 mb-4">
                <div className="text-8xl">{selectedBugData.emoji}</div>
                <div>
                  <h2 className="text-4xl font-serif font-bold text-earth-900 mb-2">
                    {selectedBugData.name}
                  </h2>
                  <p className="text-lg text-earth-600 italic">{selectedBugData.scientific}</p>
                  <div className={`mt-3 px-4 py-2 rounded-full inline-block border-2 ${getDangerColor(selectedBugData.danger)}`}>
                    <span className="font-bold">Danger Level: {selectedBugData.danger}</span>
                  </div>
                </div>
              </div>
              <p className="text-lg text-earth-700">{selectedBugData.description}</p>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Identification */}
              <div>
                <h3 className="text-2xl font-serif font-bold text-forest-800 mb-4 flex items-center gap-2">
                  <span className="text-3xl">🔍</span>
                  How to Identify
                </h3>
                <ul className="space-y-2">
                  {selectedBugData.identification.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-earth-700">
                      <span className="text-forest-600 mt-1">✓</span>
                      <span className="text-lg">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Bite Appearance */}
              <div className="bg-orange-50 rounded-lg p-6 border-2 border-orange-200">
                <h3 className="text-2xl font-serif font-bold text-orange-900 mb-3 flex items-center gap-2">
                  <span className="text-3xl">🩹</span>
                  What Bites Look Like
                </h3>
                <p className="text-lg text-orange-800">{selectedBugData.biteAppearance}</p>
              </div>

              {/* Diseases */}
              <div className="bg-red-50 rounded-lg p-6 border-2 border-red-200">
                <h3 className="text-2xl font-serif font-bold text-red-900 mb-4 flex items-center gap-2">
                  <span className="text-3xl">🏥</span>
                  Health Risks & Diseases
                </h3>
                <div className="grid md:grid-cols-2 gap-2">
                  {selectedBugData.diseases.map((disease, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-red-800">
                      <span className="mt-1">•</span>
                      <span className="text-lg">{disease}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Habitat & Season */}
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-earth-50 rounded-lg p-6 border-2 border-earth-200">
                  <h3 className="text-xl font-bold text-earth-900 mb-3 flex items-center gap-2">
                    <span className="text-2xl">🏡</span>
                    Where They Live
                  </h3>
                  <p className="text-earth-700 text-lg">{selectedBugData.habitat}</p>
                </div>
                <div className="bg-earth-50 rounded-lg p-6 border-2 border-earth-200">
                  <h3 className="text-xl font-bold text-earth-900 mb-3 flex items-center gap-2">
                    <span className="text-2xl">📅</span>
                    When They're Active
                  </h3>
                  <p className="text-earth-700 text-lg">{selectedBugData.season}</p>
                </div>
              </div>

              {/* Prevention */}
              <div className="bg-green-50 rounded-lg p-6 border-2 border-green-200">
                <h3 className="text-2xl font-serif font-bold text-green-900 mb-4 flex items-center gap-2">
                  <span className="text-3xl">🛡️</span>
                  Prevention Tips
                </h3>
                <div className="grid md:grid-cols-2 gap-3">
                  {selectedBugData.prevention.map((tip, idx) => (
                    <div key={idx} className="flex items-start gap-3 text-green-800">
                      <span className="text-green-600 mt-1">✓</span>
                      <span className="text-lg">{tip}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Treatment */}
              <div className="bg-blue-50 rounded-lg p-6 border-2 border-blue-200">
                <h3 className="text-2xl font-serif font-bold text-blue-900 mb-3 flex items-center gap-2">
                  <span className="text-3xl">💊</span>
                  Treatment & First Aid
                </h3>
                <p className="text-lg text-blue-800">{selectedBugData.treatment}</p>
              </div>
            </div>
          </div>
        )}

        {!selectedBug && (
          <div className="bg-forest-50 rounded-xl p-12 text-center border-2 border-forest-200">
            <div className="text-6xl mb-4">🐛</div>
            <h3 className="text-2xl font-serif font-bold text-earth-900 mb-2">
              Select a Bug to Learn More
            </h3>
            <p className="text-earth-600 text-lg">
              Click any bug above to see detailed information about identification, prevention, and treatment
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BugEducation;
