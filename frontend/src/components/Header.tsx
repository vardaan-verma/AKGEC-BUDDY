import { Button } from "@/components/ui/button";
import { Phone, Menu, X } from "lucide-react";
import { useState } from "react";

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    "ABOUT", "DEPARTMENTS", "ACADEMICS", "ADMISSIONS", 
    "R&D", "LIFE@AKGEC", "ACHIEVEMENTS", "PLACEMENTS", "VE CELL", "IQAC"
  ];

  return (
    <header className="w-full bg-background shadow-sm border-b border-border">
      {/* Top Bar */}
      <div className="bg-primary text-primary-foreground py-2">
        <div className="container mx-auto px-4 flex justify-end items-center">
          <div className="flex items-center gap-4 text-sm">
            <span className="bg-accent text-accent-foreground px-3 py-1 rounded font-semibold">
              AKGEC ERP Login
            </span>
            <span className="bg-accent text-accent-foreground px-3 py-1 rounded font-semibold">
              Admission Helpline
            </span>
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4" />
              <span className="font-semibold">8744052891-93</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Header */}
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and College Name */}
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 bg-gradient-to-br from-accent to-primary rounded-full flex items-center justify-center">
              <div className="w-12 h-12 bg-background rounded-full flex items-center justify-center">
                <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-full"></div>
              </div>
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-primary">
                AJAY KUMAR GARG ENGINEERING COLLEGE
              </h1>
              <p className="text-sm text-muted-foreground">
                (Affiliated to Dr. APJ Abdul Kalam Technical University, Lucknow, UP, College Code - 027)
              </p>
            </div>
          </div>

          {/* Accreditation Badges */}
          <div className="hidden lg:flex items-center gap-2">
            <div className="bg-accent text-accent-foreground px-3 py-2 rounded text-sm font-bold">
              NAAC GRADE A++
            </div>
            <div className="bg-accent text-accent-foreground px-3 py-2 rounded text-sm font-bold">
              QS I-GAUGE DIAMOND
            </div>
            <div className="bg-success text-success-foreground px-3 py-2 rounded text-sm font-bold">
              NBA ACCREDITATION
            </div>
            <div className="bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-bold">
              INSTITUTION'S INNOVATION COUNCIL
            </div>
          </div>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          >
            {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="bg-primary text-primary-foreground">
        <div className="container mx-auto px-4">
          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center justify-center">
            {navItems.map((item, index) => (
              <Button
                key={index}
                variant="ghost"
                className="text-primary-foreground hover:bg-primary-hover hover:text-white px-4 py-6 rounded-none border-r border-primary-hover last:border-r-0"
              >
                {item}
              </Button>
            ))}
          </div>

          {/* Mobile Navigation */}
          {isMobileMenuOpen && (
            <div className="lg:hidden py-4 space-y-2">
              {navItems.map((item, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  className="w-full text-left text-primary-foreground hover:bg-primary-hover justify-start"
                >
                  {item}
                </Button>
              ))}
            </div>
          )}
        </div>
      </nav>
    </header>
  );
};

export default Header;