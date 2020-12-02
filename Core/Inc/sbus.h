/*
 * sbus.h
 *
 *  Created on: Nov 26, 2019
 *      Author: Đặng Lâm Tùng
 */

typedef struct{
	uint16_t esc_value1;
	uint16_t esc_value2;
	uint16_t esc_value3;
	uint16_t esc_value4;
	uint16_t crc;
}ESC_value;




ESC_value sbus_decode(uint8_t data[7]){
	ESC_value value;
	value.esc_value1 = (data[0] << 3) | ((data[1] & 0b11100000)>>5);
	value.esc_value2 = ((data[1] & 0b00011111)<<6)|((data[2] & 0b11111100)>>2);
	value.esc_value3 = (((data[2] &0b00000011)<<9)|(data[3]<<1))|((data[4] & 0b10000000)>>7);
	value.esc_value4 = ((data[4] & 0b01111111)<<4)|(data[5])>>4;
    value.crc = data[6];
	return value;
}
uint8_t check_CRC(ESC_value value){
    uint16_t check = value.esc_value1 + value.esc_value2 + value.esc_value3 + value.esc_value4;
    if(check % 37 != value.crc)
    	return 0;
    return 1;
}

uint8_t CRC_thurst(ESC_value value){

    if((value.esc_value1 == value.esc_value2) && (value.esc_value3 == value.esc_value4) && (value.esc_value1 == value.esc_value4))
    	return 1;
    return 0;
}
