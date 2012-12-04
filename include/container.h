#ifndef CONTAINER_H
#define CONTAINER_H

class container
{
 public:
  
  container();
  virtual ~container();
  
  double p1[17];
  double p2[17];
  double p3[17];

  double y1[17];
  double y2[17];
  double y3[17];

  double phi1[17];
  double phi2[17];
  double phi3[17];
  
  int nparton[17];

  double weight[17]; //weights of each event.
  int nevent; //number of events;

};

#endif
