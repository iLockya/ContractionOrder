#include <Adafruit_GFX.h>    // Core graphics library
#include <Adafruit_ST7735.h> // Hardware-specific library for ST7735
#include <Adafruit_ST7789.h> // Hardware-specific library for ST7789
#include <SPI.h>

#if defined(ARDUINO_FEATHER_ESP32) // Feather Huzzah32
  #define TFT_CS         14
  #define TFT_RST        15
  #define TFT_DC         32

#elif defined(ESP8266)
  #define TFT_CS         4
  #define TFT_RST        16                                            
  #define TFT_DC         5

#else
  // For the breakout board, you can use any 2 or 3 pins.
  // These pins will also work for the 1.8" TFT shield.
  #define TFT_CS        10
  #define TFT_RST        9 // Or set to -1 and connect to Arduino RESET pin
  #define TFT_DC         8
#endif

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

void setup(void){
    tft.initR(INITR_144GREENTAB);   // initialize a ST7735S chip, black tab
    tft.fillScreen(ST7735_BLACK);   // set the screen to black
    drawBoxes();
    delay(1000);
    drawLines();
    delay(1000);
    drawChinese();
    delay(1000);
}

void loop(void){
    tft.invertDisplay(true);
    delay(1000);
    tft.invertDisplay(false);
    delay(1000);
}

// 画一组嵌套的方框
void drawBoxes(void) {
  tft.fillScreen(ST7735_BLACK);
  tft.drawRect(10, 10,  40,  40, ST7735_GREEN);
  tft.drawRect(20, 30,  60,  60, ST7735_YELLOW);
  tft.drawRect(30, 50,  90,  90, ST7735_RED);
  tft.drawRect(40, 70, 120, 120, ST7735_BLUE);
  tft.drawRect(50, 90, 150, 150, ST7735_MAGENTA);
  tft.drawRect(60, 110, 180, 180, ST7735_CYAN);
  tft.drawRect(70, 130, 210, 210, ST7735_WHITE);
  tft.drawRect(80, 150, 240, 240, ST7735_BLACK);
}
// 画一组黄色的米字线+4红球
void drawLines(void) {
  tft.fillScreen(ST7735_BLACK);
  tft.drawLine(  0,   0,  239,   0, ST7735_YELLOW);
  tft.drawLine(  0,   0,   0, 159, ST7735_YELLOW);
  tft.drawLine(239,   0, 239, 159, ST7735_YELLOW);
  tft.drawLine(  0, 159, 239, 159, ST7735_YELLOW);
  tft.drawCircle(120, 120,   0, ST7735_RED);
  tft.drawCircle( 70,  70,   0, ST7735_RED);
  tft.drawCircle(170,  70,   0, ST7735_RED);
  tft.drawCircle( 70, 120,   0, ST7735_RED);
  tft.drawCircle(170, 120,   0, ST7735_RED);
  tft.drawCircle(120, 170,   0, ST7735_RED);
}

//显示较大的汉字：科
void drawChinese(void) {
  tft.setCursor(0, 0);
  tft.setTextSize(2);
  tft.setTextColor(ST7735_WHITE);
  tft.setTextWrap(true);
  tft.print("科");
  tft.setCursor(0, 20);
  tft.print("科");
  tft.setCursor(0, 40);
  tft.print("科");
  tft.setCursor(0, 60);
  tft.print("科");
  tft.setCursor(0, 80);
  tft.print("科");
  tft.setCursor(0, 100);
  tft.print("科");
  tft.setCursor(0, 120);
  tft.print("科");
  tft.setCursor(0, 140);
  tft.print("科");
  tft.setCursor(0, 160);
  tft.print("科");
  tft.setCursor(0, 180);
  tft.print("科");
  tft.setCursor(0, 200);
  tft.print("科");
  tft.setCursor(0, 220);
  tft.print("科");
  tft.setCursor(0, 240);
  tft.print("科");
  tft.setCursor(0, 260);
  tft.print("科");
  tft.setCursor(0, 280);
  tft.print("科");
  tft.setCursor(0, 300);
  tft.print("科");
  tft.setCursor(0, 320);
  tft.print("科");
  tft.setCursor(0, 340);
  tft.print("科");
  tft.setCursor(0, 360);
  tft.print("科");  