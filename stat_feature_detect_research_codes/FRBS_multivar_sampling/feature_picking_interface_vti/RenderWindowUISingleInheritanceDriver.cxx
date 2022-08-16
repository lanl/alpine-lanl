#include <QApplication>
#include <RenderWindowUISingleInheritance.h>
 
int main( int argc, char** argv )
{

  QApplication app( argc, argv );
  app.setStyle("cleanlooks");
  RenderWindowUISingleInheritance renderWindowUISingleInheritance;
  renderWindowUISingleInheritance.setWindowTitle("Interactive Training Data Selection Interface");
  renderWindowUISingleInheritance.show();

  // do some off-screen rendering, the widget has never been made visible
  // makeCurrent (); // ABSOLUTELY CRUCIAL!
  // renderOffScreen ();
 
  return app.exec();
}
