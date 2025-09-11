import { useState, useEffect } from "react";

const HeroSection = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const meritListStudents = [
    {
      name: "Shiv Prakash Singh",
      image: "/lovable-uploads/b9219e8a-1c17-4986-918c-256f3c29a6e2.png"
    },
    {
      name: "Vandit Mishra", 
      image: "/lovable-uploads/b9219e8a-1c17-4986-918c-256f3c29a6e2.png"
    },
    {
      name: "Aaradhya",
      image: "/lovable-uploads/b9219e8a-1c17-4986-918c-256f3c29a6e2.png"
    },
    {
      name: "Madhur Vikram Singh",
      image: "/lovable-uploads/b9219e8a-1c17-4986-918c-256f3c29a6e2.png"
    },
    {
      name: "Kanika Chaudhary",
      image: "/lovable-uploads/b9219e8a-1c17-4986-918c-256f3c29a6e2.png"
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % meritListStudents.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative bg-gradient-to-b from-background to-secondary/30 py-16">
      {/* College Logos Row */}
      <div className="container mx-auto px-4 mb-12">
        <div className="flex flex-wrap items-center justify-center gap-8 opacity-90">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="w-24 h-16 bg-gradient-to-br from-primary to-accent rounded flex items-center justify-center">
              <span className="text-white text-xs font-bold text-center">AJAY KUMAR GARG<br/>ENGINEERING COLLEGE</span>
            </div>
          </div>
          <div className="bg-accent/10 p-4 rounded-lg">
            <div className="w-24 h-16 bg-gradient-to-br from-accent to-primary rounded flex items-center justify-center">
              <span className="text-white text-xs font-bold text-center">NAAC<br/>GRADE A++</span>
            </div>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="w-24 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded flex items-center justify-center">
              <span className="text-white text-xs font-bold text-center">NBA<br/>ACCREDITATION</span>
            </div>
          </div>
          <div className="bg-accent/10 p-4 rounded-lg">
            <div className="w-24 h-16 bg-gradient-to-br from-accent to-orange-600 rounded flex items-center justify-center">
              <span className="text-white text-xs font-bold text-center">QS I-GAUGE<br/>DIAMOND</span>
            </div>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="w-24 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded flex items-center justify-center">
              <span className="text-white text-xs font-bold text-center">INSTITUTION'S<br/>INNOVATION<br/>COUNCIL</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 text-center">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-5xl md:text-6xl font-bold mb-6 text-gradient" style={{
            fontFamily: 'serif',
            background: 'linear-gradient(45deg, #8B4513, #DAA520)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}>
            Heartiest Congratulations
          </h2>
          
          <h3 className="text-4xl md:text-5xl font-bold text-primary mb-12">
            AKTU MERIT LIST 2025
          </h3>

          {/* Students Carousel */}
          <div className="relative h-80 overflow-hidden">
            <div 
              className="flex transition-transform duration-1000 ease-in-out"
              style={{ transform: `translateX(-${currentSlide * 100}%)` }}
            >
              {meritListStudents.map((student, index) => (
                <div key={index} className="w-full flex-shrink-0 flex justify-center">
                  <div className="flex flex-col items-center fade-in-up">
                    {/* Laurel Wreath Container */}
                    <div className="relative">
                      {/* Golden Laurel Wreath SVG Background */}
                      <div className="absolute inset-0 scale-125 opacity-80">
                        <svg className="w-48 h-48 text-accent" viewBox="0 0 200 200" fill="currentColor">
                          <path d="M100 20 C120 30, 140 50, 150 80 C160 110, 150 140, 130 160 C110 180, 90 180, 70 160 C50 140, 40 110, 50 80 C60 50, 80 30, 100 20 Z" opacity="0.3"/>
                          {/* Simplified laurel pattern */}
                          <circle cx="100" cy="100" r="75" fill="none" stroke="currentColor" strokeWidth="8" opacity="0.6"/>
                          <circle cx="100" cy="100" r="85" fill="none" stroke="currentColor" strokeWidth="4" opacity="0.8"/>
                        </svg>
                      </div>
                      
                      {/* Student Photo */}
                      <div className="relative z-10 w-32 h-32 rounded-full overflow-hidden border-4 border-accent shadow-xl mx-auto">
                        <img 
                          src={student.image} 
                          alt={student.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>
                    
                    <h4 className="text-xl font-bold text-primary mt-6 bg-white/80 px-4 py-2 rounded-lg shadow-sm">
                      {student.name}
                    </h4>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Carousel Indicators */}
          <div className="flex justify-center space-x-2 mt-8">
            {meritListStudents.map((_, index) => (
              <button
                key={index}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  index === currentSlide ? 'bg-accent scale-125' : 'bg-muted-foreground/30'
                }`}
                onClick={() => setCurrentSlide(index)}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;