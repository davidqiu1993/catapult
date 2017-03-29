#!/usr/bin/python
#\file    dynamixel_lib.py
#\brief   Library to control a Dynamixel servo
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.28, 2017

#This code is based on DynamixelSDK/python/protocol2_0/sync_read_write.py
#DynamixelSDK: https://github.com/ROBOTIS-GIT/DynamixelSDK
#Dynamixel XM430-W350: http://support.robotis.com/en/product/actuator/dynamixel_x/xm_series/xm430-w350.htm

import dynamixel_functions as dynamixel                     # Uses Dynamixel SDK library


class TDynamixel1:
  def __init__(self):
    # For Dynamixel PRO 54-200 with USB2DYNAMIXEL
    #self.ADDR_PRO_TORQUE_ENABLE       = 562
    #self.ADDR_PRO_GOAL_POSITION       = 596
    #self.ADDR_PRO_PRESENT_POSITION    = 611
    # For Dynamixel XM430-W350 with USB2DYNAMIXEL
    self.ADDR_PRO_TORQUE_ENABLE       = 64
    self.ADDR_PRO_GOAL_POSITION       = 116
    self.ADDR_PRO_PRESENT_POSITION    = 132
    self.ADDR_OPERATING_MODE          = 11

    self.OP_MODE_POSITION            = 3                             # Position control mode

    # Protocol version
    self.PROTOCOL_VERSION            = 2                             # See which protocol version is used in the Dynamixel

    # Default setting
    self.DXL_ID                      = 1                             # Dynamixel ID: 1
    self.BAUDRATE                    = 57600
    self.DEVICENAME                  = '/dev/ttyUSB0'                # Check which port is being used on your controller
                                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0"

    self.TORQUE_ENABLE               = 1                             # Value for enabling the torque
    self.TORQUE_DISABLE              = 0                             # Value for disabling the torque
    self.DXL_MINIMUM_POSITION_VALUE  = 0                       # Dynamixel will rotate between this value
    self.DXL_MAXIMUM_POSITION_VALUE  = 4095                        # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)

    self.COMM_SUCCESS                = 0                             # Communication Success result value
    self.COMM_TX_FAIL                = -1001                         # Communication Tx Failed

    self.port_num = None

  def __del__(self):
    if self.port_num is None:  self.Quit()

  def Setup(self):
    # Initialize PortHandler Structs
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    self.port_num = dynamixel.portHandler(self.DEVICENAME)

    # Initialize PacketHandler Structs
    dynamixel.packetHandler()

    #Open port
    if dynamixel.openPort(self.port_num):
      print 'Opened a port:', self.DEVICENAME
    else:
      print 'Failed to open a port:', self.DEVICENAME
      self.port_num = None
      return False

    #Set port baudrate
    if dynamixel.setBaudRate(self.port_num, self.BAUDRATE):
      print 'Changed the baud rate to:',self.BAUDRATE
    else:
      print 'Failed to change the baud rate to:',self.BAUDRATE
      return False

    self.SetOpMode(self.OP_MODE_POSITION)
    self.EnableTorque()

  def Quit(self):
    self.DisableTorque()
    # Close port
    dynamixel.closePort(self.port_num)
    self.port_num = None

  #Check the result of sending a command, and print an error message
  def CheckTxRxResult(self):
    if dynamixel.getLastTxRxResult(self.port_num, self.PROTOCOL_VERSION) != self.COMM_SUCCESS:
      dynamixel.printTxRxResult(self.PROTOCOL_VERSION, dynamixel.getLastTxRxResult(self.port_num, self.PROTOCOL_VERSION))
      return False
    elif dynamixel.getLastRxPacketError(self.port_num, self.PROTOCOL_VERSION) != 0:
      dynamixel.printRxPacketError(self.PROTOCOL_VERSION, dynamixel.getLastRxPacketError(self.port_num, self.PROTOCOL_VERSION))
      return False
    return True

  #Changing operating mode
  def SetOpMode(self, mode):
    self.DisableTorque()  #NOTE: We need to disable before changing the operating mode
    #print 'Operating mode:', dynamixel.read1ByteTxRx(self.port_num, self.PROTOCOL_VERSION, self.DXL_ID, self.ADDR_OPERATING_MODE)
    dynamixel.write1ByteTxRx(self.port_num, self.PROTOCOL_VERSION, self.DXL_ID, self.ADDR_OPERATING_MODE, mode)
    self.CheckTxRxResult()

  #Enable Dynamixel Torque
  def EnableTorque(self, enabled=True):
    value = self.TORQUE_ENABLE if enabled else self.TORQUE_DISABLE
    dynamixel.write1ByteTxRx(self.port_num, self.PROTOCOL_VERSION, self.DXL_ID, self.ADDR_PRO_TORQUE_ENABLE, value)
    self.CheckTxRxResult()

  #Disable Dynamixel Torque
  def DisableTorque(self):
    self.EnableTorque(enabled=False)

  #Get current position
  def Position(self):
    position = dynamixel.read4ByteTxRx(self.port_num, self.PROTOCOL_VERSION, self.DXL_ID, self.ADDR_PRO_PRESENT_POSITION)
    if not self.CheckTxRxResult():  return None
    return position

  #Move the position to a given value.
  #  target: Target position, should be in [self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE]
  #  wait:   True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveTo(self, target, wait=True, threshold=20):
    target = int(target)
    if target < self.DXL_MINIMUM_POSITION_VALUE:  target = self.DXL_MINIMUM_POSITION_VALUE
    elif target > self.DXL_MAXIMUM_POSITION_VALUE:  target = self.DXL_MAXIMUM_POSITION_VALUE

    # Write goal position
    dynamixel.write4ByteTxRx(self.port_num, self.PROTOCOL_VERSION, self.DXL_ID, self.ADDR_PRO_GOAL_POSITION, target)
    if not self.CheckTxRxResult():  return

    while wait:
      pos = self.Position()
      if pos is None:  return
      #print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.DXL_ID, target, pos))
      if not (abs(target - pos) > threshold):  break


if __name__=='__main__':
  import os
  import sys, tty, termios
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  def getch():
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

  dxl = TDynamixel1()
  dxl.Setup()

  index = 0
  positions = [dxl.DXL_MINIMUM_POSITION_VALUE, dxl.DXL_MAXIMUM_POSITION_VALUE]
  while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
      break

    dxl.MoveTo(positions[index])
    print 'Current position=',dxl.Position()
    index = 1 if index==0 else 0


