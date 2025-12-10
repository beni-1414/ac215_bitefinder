import React, { useState, useRef, useEffect } from 'react';

interface UploadSectionProps {
  onAnalyze: (image: string, notes: string) => void;
  isAnalyzing: boolean;
  initialImage?: string | null;
  initialNotes?: string;
  onImageChange?: (dataUrl: string | null) => void;
  onNotesChange?: (notes: string) => void;
}

function resizeImageToDataUrl(
  file: File,
  options: { maxWidth: number; maxHeight: number; quality: number } = {
    maxWidth: 1024,
    maxHeight: 1024,
    quality: 0.8,
  }
): Promise<string> {
  const { maxWidth, maxHeight, quality } = options;

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;

        const ratio = Math.min(
          maxWidth / width,
          maxHeight / height,
          1 // never upscale
        );
        width = width * ratio;
        height = height * ratio;

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Could not get 2D context"));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);

        // Export as JPEG (works great for iPhone photos)
        const dataUrl = canvas.toDataURL("image/jpeg", quality);
        resolve(dataUrl);
      };
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = e.target?.result as string;
    };

    reader.readAsDataURL(file);
  });
}


const UploadSection: React.FC<UploadSectionProps> = ({
  onAnalyze,
  isAnalyzing,
  initialImage = null,
  initialNotes = '',
  onImageChange,
  onNotesChange
}) => {
  const [preview, setPreview] = useState<string | null>(initialImage);
  const [notes, setNotes] = useState(initialNotes);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setPreview(initialImage || null);
  }, [initialImage]);

  useEffect(() => {
    setNotes(initialNotes || '');
  }, [initialNotes]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = async (file: File) => {
    try {
      const resizedDataUrl = await resizeImageToDataUrl(file, {
        maxWidth: 1024,
        maxHeight: 1024,
        quality: 0.8,
      });

      setPreview(resizedDataUrl);
      onImageChange?.(resizedDataUrl);
    } catch (err) {
      console.error("Error processing image", err);
      // Optional: simple user-facing message
      alert("Sorry, this image format or file is not supported in this browser.");
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (preview) {
      // Extract base64 raw string (remove data:image/jpeg;base64, prefix)
      const base64Data = preview.split(',')[1];
      onAnalyze(base64Data, notes);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-xl shadow-xl overflow-hidden border border-earth-200">
      <div className="p-6 space-y-6">
        {/* Image Upload Area */}
        <div
          onClick={triggerFileSelect}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300
            ${preview ? 'border-forest-500 bg-forest-50' : 'border-earth-300 hover:border-forest-400 hover:bg-earth-50'}
          `}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            ref={fileInputRef}
          />

          {preview ? (
            <div className="relative group">
              <img
                src={preview}
                alt="Upload preview"
                className="max-h-64 mx-auto rounded shadow-md"
              />
              <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded">
                <span className="text-white font-medium">Click to change photo</span>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="mx-auto w-16 h-16 bg-earth-200 rounded-full flex items-center justify-center text-earth-600">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z" />
                </svg>
              </div>
              <div>
                <p className="text-earth-900 font-medium">Upload a photo</p>
                <p className="text-earth-500 text-sm">JPG, PNG up to 5MB</p>
              </div>
            </div>
          )}
        </div>

        {/* Text Input */}
        <div>
          <label htmlFor="notes" className="block text-sm font-medium text-earth-800 mb-1">
            Tell us what you know and how you feel.
          </label>
          <textarea
            id="notes"
            rows={3}
            className="w-full rounded-md border-earth-300 shadow-sm focus:border-forest-500 focus:ring-forest-500 bg-earth-50 p-3 text-earth-900"
            placeholder="e.g., Hiking by a stream when I felt a sting on my ankle. Now there's a red, itchy bump and some swelling..."
            value={notes}
            onChange={(e) => {
              setNotes(e.target.value);
              onNotesChange?.(e.target.value);
            }}
          />
        </div>

        {/* Action Button */}
        <button
          onClick={handleSubmit}
          disabled={isAnalyzing}
          className={`
            w-full flex items-center justify-center py-4 px-6 border border-transparent rounded-lg shadow-sm text-base font-medium text-white
            transition-all duration-200
            ${isAnalyzing
              ? 'bg-earth-300 cursor-not-allowed'
              : 'bg-forest-700 hover:bg-forest-800 shadow-forest-900/20 shadow-lg hover:-translate-y-0.5'
            }
          `}
        >
          {isAnalyzing ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </>
          ) : (
            'Identify Bug'
          )}
        </button>
      </div>
    </div>
  );
};

export default UploadSection;
