import { Button } from "@/components/ui/button";
import { MapPin, Phone, Mail, Facebook, Twitter, Linkedin, Instagram } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-primary text-primary-foreground">
      {/* Main Footer Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* College Info */}
          <div className="space-y-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-accent to-primary-hover rounded-full flex items-center justify-center">
                <div className="w-8 h-8 bg-primary-foreground rounded-full flex items-center justify-center">
                  <div className="w-4 h-4 bg-gradient-to-br from-primary to-accent rounded-full"></div>
                </div>
              </div>
              <div>
                <h3 className="font-bold text-lg">AKGEC</h3>
                <p className="text-sm opacity-80">Excellence in Education</p>
              </div>
            </div>
            <p className="text-sm opacity-90 leading-relaxed">
              Ajay Kumar Garg Engineering College is committed to providing quality technical education 
              and fostering innovation among students to meet industry demands.
            </p>
            <div className="flex space-x-3">
              <Button variant="ghost" size="icon" className="text-primary-foreground hover:bg-primary-hover">
                <Facebook className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-primary-foreground hover:bg-primary-hover">
                <Twitter className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-primary-foreground hover:bg-primary-hover">
                <Linkedin className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-primary-foreground hover:bg-primary-hover">
                <Instagram className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Quick Links</h4>
            <div className="space-y-2">
              {['About Us', 'Admissions', 'Academic Programs', 'Faculty', 'Research', 'Campus Life', 'Placements', 'Alumni'].map((link) => (
                <Button key={link} variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent justify-start">
                  {link}
                </Button>
              ))}
            </div>
          </div>

          {/* Academic Programs */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Programs</h4>
            <div className="space-y-2">
              {['B.Tech CSE', 'B.Tech ECE', 'B.Tech ME', 'B.Tech CE', 'M.Tech', 'MBA', 'MCA', 'Ph.D Programs'].map((program) => (
                <Button key={program} variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent justify-start">
                  {program}
                </Button>
              ))}
            </div>
          </div>

          {/* Contact Info */}
          <div>
            <h4 className="font-semibold text-lg mb-4">Contact Us</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <MapPin className="w-5 h-5 mt-0.5 text-accent" />
                <div className="text-sm">
                  <p>27th Km Milestone,</p>
                  <p>Delhi - Meerut Expressway,</p>
                  <p>P.O. Adhyatmik Nagar,</p>
                  <p>Ghaziabad - 201015</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Phone className="w-5 h-5 text-accent" />
                <div className="text-sm">
                  <p>+91-8744052891-93</p>
                  <p>0120-2441481</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Mail className="w-5 h-5 text-accent" />
                <div className="text-sm">
                  <p>info@akgec.ac.in</p>
                  <p>admissions@akgec.ac.in</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-primary-hover bg-primary-hover/20">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm opacity-80 mb-4 md:mb-0">
              <p>© 2025 Ajay Kumar Garg Engineering College. All rights reserved.</p>
            </div>
            <div className="flex flex-wrap gap-4 text-sm">
              <Button variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent">
                Privacy Policy
              </Button>
              <Button variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent">
                Terms of Service
              </Button>
              <Button variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent">
                Site Map
              </Button>
              <Button variant="ghost" className="p-0 h-auto text-primary-foreground hover:text-accent">
                Disclaimer
              </Button>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;