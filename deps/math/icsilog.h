#ifndef __ICSI_LOG__
#define __ICSI_LOG__

////////////////////////////////////////////////////////////////////////////////////
// ICSIlog Copyright taken from ICSIlog source                                    //
// Copyright (C) 2007 International Computer Science Institute                    //
// 1947 Center Street, Suite 600                                                  //
// Berkeley, CA 94704                                                             //
//                                                                                //
// Contact information:                                                           //
//    Oriol Vinyals	vinyals@icsi.berkeley.edu                                 //
//    Gerald Friedland 	fractor@icsi.berkeley.edu                                 //
//                                                                                //
// This program is free software; you can redistribute it and/or modify           //
// it under the terms of the GNU General Public License as published by           //
// the Free Software Foundation; either version 2 of the License, or              //
// (at your option) any later version.                                            //
//                                                                                //
// This program is distributed in the hope that it will be useful,                //
// but WITHOUT ANY WARRANTY; without even the implied warranty of                 //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  //
// GNU General Public License for more details.                                   //
//                                                                                //
// You should have received a copy of the GNU General Public License along        //
// with this program; if not, write to the Free Software Foundation, Inc.,        //
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.                    //
//                                                                                //
// Authors                                                                        //
// -------                                                                        //
//                                                                                //
// Oriol Vinyals	vinyals@icsi.berkeley.edu                                 //
// Gerald Friedland 	fractor@icsi.berkeley.edu                                 //
//                                                                                //
// Acknowledgements                                                               //
// ----------------                                                               //
//                                                                                //
// Thanks to Harrison Ainsworth (hxa7241@gmail.com) for his idea that             //
// doubled the accuracy.                                                          //
////////////////////////////////////////////////////////////////////////////////////

// ICSIlog V 2.0
const std::vector<float> fill_icsi_log_table2(const unsigned int precision) {
  std::vector<float> pTable(static_cast<size_t>(pow(2,precision)));

  // step along table elements and x-axis positions
  // (start with extra half increment, so the steps intersect at their midpoints.)
  float oneToTwo = 1.0f + (1.0f / (float)( 1 <<(precision + 1) ));
  for(int i = 0;  i < (1 << precision);  ++i ) {
    // make y-axis value for table element
    pTable[i] = logf(oneToTwo) / 0.69314718055995f;
    oneToTwo += 1.0f / (float)( 1 << precision );
  }
  return pTable;
}

// ICSIlog v2.0
inline float icsi_log(const float val) {
  const unsigned int precision(10);
  static std::vector<float> pTable = fill_icsi_log_table2(precision);

  // get access to float bits
  static_assert(sizeof(int)==sizeof(float),"int and float are not the same size.");
  const int* const pVal = reinterpret_cast<const int*>(&val);

  // extract exponent and mantissa (quantized)
  const int exp = ((*pVal >> 23) & 255) - 127;
  const int man = (*pVal & 0x7FFFFF) >> (23 - precision);

  // exponent plus lookup refinement
  return ((float)(exp) + pTable[man]) * 0.69314718055995f;
}

#endif
