

void setup() {
  // put your setup code here, to run once:
  //pinMode(10,OUTPUT);
  pinMode(6,OUTPUT);
  pinMode(9,OUTPUT);
  pinMode(3,OUTPUT);
  pinMode(5,OUTPUT);
  Serial.begin(9600);

  digitalWrite(3,HIGH);
  digitalWrite(5,HIGH);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  analogWrite(9,0);
  for(int x = 0; x < 255; x++){
    analogWrite(6,x);
    delay(10);
  }
  //analogWrite(10,70);
  Serial.println("switch");
  delay(5000);
  analogWrite(6,0);
  for(int x = 0; x < 255; x++){
    analogWrite(9,x);
    delay(10);
  }
  //analogWrite(10,0);
  Serial.println("switch");
  delay(5000);
  

}
