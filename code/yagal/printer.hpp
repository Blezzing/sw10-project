#pragma once

#include <iostream>
#include <string>

namespace yagal::printer {
    
    // A pretty printer for convenience.
    // Silent: Nothing
    // Standard: error
    // Verbose: error and Info
    // Debug: error, info, debug
    class Printer {
    public:
        enum class Mode { Silent, Standard, Verbose, Debug };

    private:
        std::string _prefix;
        Mode _mode;
        // To prevent printing, e.g. in the case of silent
        class NullBuffer : public std::streambuf{
            public:
            int overflow(int c) {return c;}
        };
        class NullStream : public std::ostream{
            NullBuffer _nb;
            public:
            NullStream() 
            : std::ostream(&_nb){}
        };
        NullStream _nullStream;


    public:
        Printer(std::string prefix, Mode mode = Mode::Standard)
        : _prefix(prefix), _mode(mode) { }

        void setMode(Mode mode) {
            _mode = mode;
        }

        std::ostream& error() {
            if (_mode == Mode::Silent) { 
                return _nullStream;
            }
            std::cerr << "[ERROR] " << _prefix << ": ";
            return std::cerr;
        }

        std::ostream& debug() {
            if (_mode != Mode::Debug) { 
                return _nullStream;
            }
            std::cout << "[DEBUG] " << _prefix << ": ";
        }

        std::ostream& info() {
            if (_mode == Mode::Silent || _mode == Mode::Standard) { 
                return _nullStream;
            }
            std::cout << "[INFO]  " << _prefix << ": ";
        }
    };
}