import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import ChatBot from "@/components/ChatBot";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        {/* Additional content sections can be added here */}
      </main>
      <Footer />
      <ChatBot />
    </div>
  );
};

export default Index;
