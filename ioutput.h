#ifndef IOUTPUT_H
#define IOUTPUT_H

class IOutput
{
public:
    IOutput();
    virtual float get_output() const = 0;
};

#endif // IOUTPUT_H
