import torch


def pack_bits(indices: torch.Tensor, bits: int) -> bytes:
    """
    Pack indices (1D LongTensor) into a bitstream of exact length.
    Vectorized implementation using torch operations for speed.
    """
    assert indices.ndim == 1
    n = indices.numel()
    if n == 0:
        return b""
    
    # Ensure indices are on CPU and proper dtype
    indices = indices.to(dtype=torch.int64, device="cpu")
    
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    
    # Create output byte tensor
    out = torch.zeros(total_bytes, dtype=torch.uint8)
    
    # Compute bit position for each index (MSB first)
    # bit_positions[i] = starting bit position for indices[i]
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
    
    # For each bit in the index value, compute which byte and bit offset it goes to
    for bit_idx in range(bits):
        # Extract the (bits - 1 - bit_idx)-th bit from each index (MSB first)
        bit_vals = (indices >> (bits - 1 - bit_idx)) & 1
        
        # Compute global bit position for this bit
        global_bit_pos = bit_positions + bit_idx
        
        # Compute byte index and bit offset within byte
        byte_idx = global_bit_pos // 8
        bit_offset = 7 - (global_bit_pos % 8)  # MSB first within byte
        
        # Scatter add the bits to output bytes
        out.scatter_add_(0, byte_idx, (bit_vals << bit_offset).to(torch.uint8))
    
    return bytes(out.numpy())


def unpack_bits(b: bytes, n: int, bits: int) -> torch.Tensor:
    """
    Unpack exactly n indices from bitstream b.
    Vectorized implementation using torch operations for speed.
    """
    if n == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Convert bytes to tensor
    byte_tensor = torch.frombuffer(bytearray(b), dtype=torch.uint8).to(torch.int64)
    
    # Output tensor
    out = torch.zeros(n, dtype=torch.int64)
    
    # Bit positions for each index
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
    
    # For each bit in the index value
    for bit_idx in range(bits):
        # Global bit position for this bit of each index
        global_bit_pos = bit_positions + bit_idx
        
        # Byte index and bit offset
        byte_idx = global_bit_pos // 8
        bit_offset = 7 - (global_bit_pos % 8)
        
        # Extract bits from bytes
        bit_vals = (byte_tensor[byte_idx] >> bit_offset) & 1
        
        # Set the corresponding bit in output (MSB first)
        out |= bit_vals << (bits - 1 - bit_idx)
    
    return out
